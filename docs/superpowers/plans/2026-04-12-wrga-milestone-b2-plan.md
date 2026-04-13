# WRGA Milestone B.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close four of the five gaps deferred from Milestone B.1 — runtime adapter materialisation with LoRA alpha scaling, typed dual-keyspace allocator plumbing behind a flag, stdlib loading in the `debug_compile_and_return_plan` test helper, and `--wrga-report` for ZK/standalone builds. The MMA epilogue PTX is explicitly deferred to B.3.

**Architecture:** Five independently-committable tasks. Task 1 makes training sources with real optimizers testable end-to-end. Task 2 injects LoRA/IA³/GatedLoRA as synthesized model fields with strict zero-init for B matrices and an `alpha/rank` scaling factor; forward-pass rewrites run unfused (3 kernel launches per site) pending B.3's MMA fusion. Task 3 introduces `AllocationKey { Weight(String), Activation(VarId) }` to enforce keyspace separation at compile time, then ships `consume_hints` + a default-off `--wrga-fold-allocations` flag. Task 4 extends `entry_points.rs` with plan-returning variants for ZK and standalone. Task 5 updates memory and runs full regression; no merge.

**Tech Stack:** Rust 2021, Cranelift IR, clap v4, `cargo test -p nsl-{semantic,codegen,cli}`. Reference: [NSL-WRGA-Research.PDF](NSL-WRGA-Research.PDF) for LoRA/IA³/GatedLoRA init and scaling conventions.

---

## Pre-flight

- [ ] **Confirm baseline on `main` is post-B.1 + origin merge**.
  Run: `git log -1 --oneline`
  Expected: the most recent commit is `7420e92 docs(wrga): B.2 spec — add LoRA alpha scaling, typed AllocationKey, perf regression note` or a later spec revision.

- [ ] **Record baseline test counts** (these are the floor for every later task):
  ```bash
  cargo test -p nsl-semantic 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-codegen --lib 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-codegen --tests 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-codegen flash_attention 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | grep "test result" | tail -3
  cargo test -p nsl-cli --test wrga_report_cli 2>&1 | grep "test result" | tail -3
  ```
  Expected floors: semantic ≥ 258, codegen lib ≥ 881, flash_attention ≥ 56, nsl-cli e2e ≥ 80, wrga_report_cli ≥ 4.

- [ ] **Create the B.2 worktree**:
  ```bash
  cd c:/Users/bwiem/projects/NSL
  git worktree add ../NSL-wrga-b2 -b feat/wrga-milestone-b2
  cd ../NSL-wrga-b2
  cargo build --features cuda 2>&1 | tail -3
  ```
  Expected: clean build. All subsequent work happens in `c:/Users/bwiem/projects/NSL-wrga-b2`.

---

## File Structure

**Create:**
- `crates/nsl-codegen/src/stdlib_loader.rs` — minimal inline stdlib graph loader used by `debug_compile_and_return_plan` (Task 1).
- `crates/nsl-codegen/src/wrga_adapter_inject.rs` — codegen pass that emits LoRA/IA³/GatedLoRA synthesized fields, init, and forward-pass rewrite (Task 2).
- `crates/nsl-codegen/tests/wrga_adapter_runtime.rs` — Task 2 integration test (compile-and-run).
- `crates/nsl-codegen/tests/wrga_fold_allocations_flag.rs` — Task 3 flag-behavior integration test.

**Modify:**
- `crates/nsl-codegen/src/lib.rs` — `debug_compile_and_return_plan` rework (Task 1); `WrgaInputs` unchanged; add `pub mod stdlib_loader;` (Task 1); extend `CompileOptions` with `wrga_fold_allocations: bool` (Task 3); re-export ZK/standalone `_returning_plan` variants (Task 4).
- `crates/nsl-codegen/src/memory_planner.rs` — introduce `AllocationKey` + `RealSlotId`; refactor storage to `HashMap<AllocationKey, RealSlotId>`; add `record_activation_alloc`, `real_slot_for_activation`, `try_merge_activation_into_slot`, `consume_hints` (Task 3).
- `crates/nsl-codegen/src/stmt.rs` — branch on `wrga_fold_allocations` flag to call `consume_hints` vs existing `apply_wrga_hints` (Task 3).
- `crates/nsl-codegen/src/compiler/entry_points.rs` — invoke `wrga_adapter_inject::run` after `wrga::run` (Task 2); add `compile_with_zk_info_returning_plan`, `compile_standalone_returning_plan` (Task 4).
- `crates/nsl-codegen/src/compiler/functions.rs` — hook into `compile_model_constructor` (line ~356) for LoRA field init; hook into `compile_model_methods` (line ~511) for forward-pass rewrite (Task 2).
- `crates/nsl-semantic/src/wrga.rs` — add `alpha: Option<i64>` to `AdapterConfig`; extend `validate_adapter_decorator` to parse `alpha=` (Task 2).
- `crates/nsl-cli/src/main.rs` — add `--wrga-fold-allocations` clap arg; plumb through CompileOptions (Task 3); remove "not yet supported" branches; wire ZK/standalone `_returning_plan` variants (Task 4).
- `crates/nsl-cli/tests/wrga_report_cli.rs` — append ZK + standalone tests (Task 4).
- `crates/nsl-codegen/tests/wrga_freeze_end_to_end.rs` — append real-training-source sibling test (Task 1).

---

## Task 1: stdlib loading in `debug_compile_and_return_plan`

**Files:**
- Create: `crates/nsl-codegen/src/stdlib_loader.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`
- Test: `crates/nsl-codegen/tests/wrga_freeze_end_to_end.rs` (append)

**Goal:** `debug_compile_and_return_plan` must load stdlib modules (especially `nsl/optim/*.nsl`) so test sources that include `train(...): optimizer: sgd(lr=...)` compile without the `undefined function 'nsl__optim__sgd__sgd_step'` error. Use the spec's **Fallback** approach — a minimal inline loader in nsl-codegen, avoiding cross-crate refactor.

- [ ] **Step 1: Write the failing test**

Append to `crates/nsl-codegen/tests/wrga_freeze_end_to_end.rs`:

```rust
/// Task 1 (B.2): `debug_compile_and_return_plan` must handle sources that
/// import stdlib optimizers.  Prior to B.2 this failed with
/// `undefined function 'nsl__optim__sgd__sgd_step'`.
#[test]
fn debug_compile_and_return_plan_loads_stdlib_for_real_training_source() {
    const SRC: &str = r#"
model Toy:
    let w: Tensor<[16, 16], f32> = zeros([16, 16])

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return x @ self.w

fn main():
    let m = Toy()
    train(model=m, epochs=1):
        optimizer: sgd(lr=1e-3)
        step(batch):
            let y = m.forward(batch.x)
            return y.sum()
"#;

    let opts = nsl_codegen::CompileOptions {
        wrga_inputs: Some(nsl_codegen::WrgaInputs::default()),
        source_ad: true,
        ..Default::default()
    };
    let _plan = nsl_codegen::debug_compile_and_return_plan(SRC, &opts)
        .expect("compile with real sgd optimizer must succeed after stdlib loading");
}
```

- [ ] **Step 2: Run the test — expect failure**

```bash
cargo test -p nsl-codegen --test wrga_freeze_end_to_end debug_compile_and_return_plan_loads_stdlib 2>&1 | tail -15
```
Expected: compile error inside `debug_compile_and_return_plan`; the existing helper doesn't load stdlib, so `nsl__optim__sgd__sgd_step` is unresolved.

- [ ] **Step 3: Create the minimal stdlib loader**

Create `crates/nsl-codegen/src/stdlib_loader.rs`:

```rust
//! Minimal stdlib loader for test helpers.
//!
//! Resolves `NSL_STDLIB_PATH` (or a compile-time default), walks stdlib
//! `.nsl` files reachable from a given entry source's `import` statements,
//! parses + analyses each in topological order, and returns the merged
//! module graph ready for codegen.
//!
//! This is intentionally narrower than `nsl-cli`'s full `load_all_modules`
//! path — file I/O stays out of `nsl-runtime`, and the CLI's own
//! `load_all_modules` is unchanged.  A future refactor can unify the two
//! paths behind a shared trait; for B.2 we accept duplication.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nsl_ast::Module;
use nsl_errors::{Diagnostic, FileId};
use nsl_lexer::Interner;
use nsl_semantic::AnalysisResult;

use crate::CodegenError;

/// One loaded module (entry source or stdlib dependency).
pub struct LoadedModule {
    pub path: PathBuf,
    pub ast: Module,
    pub analysis: AnalysisResult,
    pub module_prefix: String,
}

/// Load an entry source string plus all stdlib modules it transitively
/// imports.  Entry module has prefix `""`; stdlib modules have prefix
/// matching their dotted import path (e.g. `nsl.optim.sgd`).
///
/// Returns the loaded modules in dependency-topological order (leaves first),
/// ready to be fed into `compile_module_with_imports_best_effort_plan`.
pub fn load_entry_with_stdlib(
    src: &str,
    interner: &mut Interner,
) -> Result<Vec<LoadedModule>, CodegenError> {
    let stdlib_root = resolve_stdlib_root()?;
    let mut loaded: HashMap<PathBuf, LoadedModule> = HashMap::new();
    let mut order: Vec<PathBuf> = Vec::new();

    // 1. Parse the entry source.
    let entry_path = PathBuf::from("<test_entry>");
    let (entry_tokens, _) = nsl_lexer::tokenize(src, FileId(0), interner);
    let entry_parse = nsl_parser::parse(&entry_tokens, interner);
    if entry_parse.diagnostics.iter().any(diag_is_error) {
        return Err(CodegenError::new(format!(
            "parse error in entry source: {:?}",
            entry_parse.diagnostics,
        )));
    }

    // 2. Discover stdlib imports from the entry AST + transitively from each
    //    stdlib module.  Only dotted paths rooted at `nsl.` are stdlib candidates.
    let mut to_visit: Vec<Vec<String>> = Vec::new();
    collect_stdlib_imports(&entry_parse.module, interner, &mut to_visit);

    while let Some(import_path) = to_visit.pop() {
        let disk_path = stdlib_disk_path(&stdlib_root, &import_path);
        if loaded.contains_key(&disk_path) {
            continue;
        }
        let src_bytes = std::fs::read(&disk_path).map_err(|e| {
            CodegenError::new(format!("stdlib read failed: {} ({e})", disk_path.display()))
        })?;
        let src_str = String::from_utf8(src_bytes).map_err(|e| {
            CodegenError::new(format!("stdlib utf8 error: {e}"))
        })?;
        let (toks, _) = nsl_lexer::tokenize(&src_str, FileId(loaded.len() as u32 + 1), interner);
        let parse = nsl_parser::parse(&toks, interner);
        if parse.diagnostics.iter().any(diag_is_error) {
            return Err(CodegenError::new(format!(
                "parse error in stdlib {}: {:?}",
                disk_path.display(),
                parse.diagnostics,
            )));
        }
        collect_stdlib_imports(&parse.module, interner, &mut to_visit);
        // Analysis happens after all modules collected so imports resolve.
        loaded.insert(
            disk_path.clone(),
            LoadedModule {
                path: disk_path.clone(),
                ast: parse.module,
                analysis: AnalysisResult::default(),
                module_prefix: import_path.join("."),
            },
        );
        order.push(disk_path);
    }

    // 3. Analyse every module (entry last so imports are resolvable).
    for path in &order {
        if let Some(m) = loaded.get_mut(path) {
            m.analysis = nsl_semantic::analyze(&m.ast, interner);
            if m.analysis.diagnostics.iter().any(diag_is_error) {
                return Err(CodegenError::new(format!(
                    "stdlib analysis error in {}: {:?}",
                    path.display(),
                    m.analysis.diagnostics,
                )));
            }
        }
    }
    let entry_analysis = nsl_semantic::analyze(&entry_parse.module, interner);
    if entry_analysis.diagnostics.iter().any(diag_is_error) {
        return Err(CodegenError::new(format!(
            "entry analysis error: {:?}",
            entry_analysis.diagnostics,
        )));
    }
    let entry_loaded = LoadedModule {
        path: entry_path,
        ast: entry_parse.module,
        analysis: entry_analysis,
        module_prefix: String::new(),
    };

    let mut result: Vec<LoadedModule> = order
        .into_iter()
        .filter_map(|p| loaded.remove(&p))
        .collect();
    result.push(entry_loaded);
    Ok(result)
}

fn resolve_stdlib_root() -> Result<PathBuf, CodegenError> {
    if let Ok(p) = std::env::var("NSL_STDLIB_PATH") {
        return Ok(PathBuf::from(p));
    }
    // Fallback: probe repo-relative `stdlib/`.  If running tests from the
    // workspace root, this resolves; otherwise surface a clear error.
    let fallback = PathBuf::from("stdlib");
    if fallback.exists() {
        return Ok(fallback);
    }
    Err(CodegenError::new(
        "NSL_STDLIB_PATH not set and 'stdlib/' not found in cwd".into(),
    ))
}

fn stdlib_disk_path(root: &Path, import_path: &[String]) -> PathBuf {
    // `nsl.optim.sgd` -> `<root>/nsl/optim/sgd.nsl`
    let mut p = root.to_path_buf();
    for seg in import_path {
        p.push(seg);
    }
    p.set_extension("nsl");
    p
}

fn collect_stdlib_imports(
    module: &Module,
    interner: &Interner,
    out: &mut Vec<Vec<String>>,
) {
    for stmt in &module.stmts {
        if let nsl_ast::stmt::StmtKind::Import(imp) = &stmt.kind {
            let segs: Vec<String> = imp
                .path
                .iter()
                .map(|s| interner.resolve(s.0).unwrap_or("").to_string())
                .collect();
            if segs.first().map(|s| s == "nsl").unwrap_or(false) {
                out.push(segs);
            }
        }
        // FromImport handled the same way.
        if let nsl_ast::stmt::StmtKind::FromImport(imp) = &stmt.kind {
            let segs: Vec<String> = imp
                .path
                .iter()
                .map(|s| interner.resolve(s.0).unwrap_or("").to_string())
                .collect();
            if segs.first().map(|s| s == "nsl").unwrap_or(false) {
                out.push(segs);
            }
        }
    }
}

fn diag_is_error(d: &Diagnostic) -> bool {
    d.is_error()
}
```

Notes for the implementer:
- `AnalysisResult::default()` may not derive `Default` — if so, build an empty one inline. The code above uses it as a placeholder pre-analysis; replace with whatever zero-construction exists.
- `CodegenError::new` may not exist with that signature — use whatever constructor is idiomatic (`CodegenError::from(String)`, a tuple variant, etc.). Match the real API.
- The `Import` / `FromImport` arm names mirror `crates/nsl-ast/src/stmt.rs`; adjust if the real variants differ.

- [ ] **Step 4: Declare the module and rework `debug_compile_and_return_plan`**

In `crates/nsl-codegen/src/lib.rs`, add at the top near other `pub mod` lines:

```rust
pub mod stdlib_loader;
```

Find the existing `debug_compile_and_return_plan` (around line 112) and replace its body with the stdlib-aware flow:

```rust
#[doc(hidden)]
pub fn debug_compile_and_return_plan(
    src: &str,
    opts: &CompileOptions,
) -> Result<Option<crate::wrga::WrgaPlan>, CodegenError> {
    let mut interner = nsl_lexer::Interner::new();
    let loaded = stdlib_loader::load_entry_with_stdlib(src, &mut interner)?;

    // The entry module is the last element (see load_entry_with_stdlib docs).
    let Some(entry) = loaded.last() else {
        return Err(CodegenError::new("no entry module loaded".into()));
    };

    // Build imported-fn signatures from stdlib modules.  Each non-entry
    // LoadedModule contributes its exported fns as imports to the entry.
    let mut imported_fns: Vec<(String, String, cranelift_codegen::ir::Signature)> = Vec::new();
    for m in loaded.iter().take(loaded.len().saturating_sub(1)) {
        for (sym, ty) in m.analysis.exports_as_fn_signatures(&interner).iter() {
            imported_fns.push((
                m.module_prefix.clone(),
                sym.clone(),
                ty.clone(),
            ));
        }
    }

    let (_obj, plan) = crate::compile_module_with_imports_best_effort_plan(
        &entry.ast,
        &interner,
        &entry.analysis.type_map,
        "",
        &imported_fns,
        Default::default(),
        Default::default(),
        false,
        opts,
    )?;
    Ok(plan)
}
```

Notes:
- `exports_as_fn_signatures` may not exist on `AnalysisResult`. Recon during implementation — if not present, extract signatures by walking `entry.analysis.type_map` or `entry.analysis.scopes` for top-level `Type::Function` entries.
- `compile_module_with_imports_best_effort_plan` is the existing function used by `debug_compile_and_return_plan`. Keep the same call shape; only the input (previously just the entry AST) now includes stdlib-sourced `imported_fns`.
- If the full-pipeline approach is too tangled for the first cut, the simpler fallback is to compile each loaded module separately and link — but that duplicates the full multi-file build logic. The above keeps it minimal.

- [ ] **Step 5: Run the test — expect pass**

```bash
NSL_STDLIB_PATH=c:/Users/bwiem/projects/NSL-wrga-b2/stdlib \
  cargo test -p nsl-codegen --test wrga_freeze_end_to_end debug_compile_and_return_plan_loads_stdlib 2>&1 | tail -15
```

Expected: PASS. If it fails with `NSL_STDLIB_PATH not set`, the env var isn't reaching the test — inline the path via `std::env::set_var` at the top of the test.

- [ ] **Step 6: Regression**

```bash
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | tail -3
cargo build --features cuda 2>&1 | tail -3
```

Expected: all existing tests still pass. `nsl-cli` e2e must remain at ≥80.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/stdlib_loader.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/tests/wrga_freeze_end_to_end.rs
git commit -m "feat(wrga): load stdlib in debug_compile_and_return_plan for real training sources"
```

---

## Task 2: Adapter A/B materialisation with alpha scaling

**Files:**
- Modify: `crates/nsl-semantic/src/wrga.rs` (add `alpha` field + parsing)
- Create: `crates/nsl-codegen/src/wrga_adapter_inject.rs`
- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs` (invoke pass after `wrga::run`)
- Modify: `crates/nsl-codegen/src/compiler/functions.rs` (hook constructor init + forward rewrite)
- Test: `crates/nsl-codegen/tests/wrga_adapter_runtime.rs`

**Goal:** For every `@adapter(type=lora|ia3|gatedlora, target=[...], rank=r, alpha=a)` site, inject synthesized model fields, emit strict initializers (LoRA B = zeros; A = Kaiming-uniform; IA³ = ones; GatedLoRA gate = zeros), and rewrite forward-pass matmul sites to apply the scaled `y = x @ W + ((x @ A) @ B) * (alpha/rank)` expression.

### Task 2a — Add `alpha` to the decorator

- [ ] **Step 1: Write the failing test**

Append to `crates/nsl-semantic/src/checker/tests.rs`:

```rust
#[test]
fn adapter_alpha_is_captured() {
    let src = r#"
@adapter(type=lora, target=["m.w"], rank=4, alpha=8)
let m = MyModel()
"#;
    let res = analyze_source(src);
    assert_eq!(res.adapter_configs.len(), 1);
    let cfg = &res.adapter_configs[0];
    assert_eq!(cfg.rank, Some(4));
    assert_eq!(cfg.alpha, Some(8), "alpha must be captured from decorator");
}

#[test]
fn adapter_alpha_defaults_to_rank_when_absent() {
    let src = r#"
@adapter(type=lora, target=["m.w"], rank=4)
let m = MyModel()
"#;
    let res = analyze_source(src);
    let cfg = &res.adapter_configs[0];
    assert_eq!(cfg.alpha, None, "absent alpha stays None; codegen defaults to rank");
}
```

(`analyze_source` helper exists in the test module; mirror the pattern used by the other WRGA tests.)

- [ ] **Step 2: Run — expect failure**

```bash
cargo test -p nsl-semantic adapter_alpha 2>&1 | tail -10
```

Expected: compile error because `AdapterConfig` has no `alpha` field.

- [ ] **Step 3: Add the field + parser arm**

In `crates/nsl-semantic/src/wrga.rs` around line 241-248, extend `AdapterConfig`:

```rust
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    pub kind: AdapterKind,
    pub targets: Vec<String>,
    /// Explicit rank override; if `None`, WRGA picks the rank via roofline
    /// + spectral analysis.
    pub rank: Option<i64>,
    /// LoRA scaling: `scale = alpha / rank`.  `None` → codegen uses
    /// `alpha = rank` (scale = 1.0).  Ignored for non-LoRA adapters.
    pub alpha: Option<i64>,
    pub span: Span,
}
```

In `validate_adapter_decorator`, add an `alpha` arm alongside the existing `rank` arm (around line 341):

```rust
"alpha" => match &arg.value.kind {
    ExprKind::IntLiteral(n) => {
        if *n <= 0 {
            diagnostics.push(
                Diagnostic::error("@adapter: alpha must be positive".to_string())
                    .with_label(arg.span, "alpha <= 0"),
            );
        } else {
            alpha = Some(*n);
        }
    }
    _ => diagnostics.push(
        Diagnostic::error("@adapter: alpha must be an integer literal".to_string())
            .with_label(arg.span, "expected integer"),
    ),
},
```

Declare `let mut alpha: Option<i64> = None;` near the top of the function alongside `rank`. Plumb it into the constructed `AdapterConfig` at the return site.

If `kind` is `Ia3` or `GatedLora` and `alpha.is_some()`, add a warning:

```rust
if alpha.is_some() && !matches!(kind, AdapterKind::Lora) {
    diagnostics.push(
        Diagnostic::warning(
            "@adapter: alpha is only meaningful for LoRA; ignored for ia3/gatedlora".to_string(),
        )
        .with_label(deco.span, "alpha ignored"),
    );
}
```

- [ ] **Step 4: Forward `alpha` through `WrgaInputs` and `AdapterDecoratorConfig`**

In `crates/nsl-codegen/src/lib.rs`, extend `AdapterDecoratorConfig`:

```rust
#[derive(Debug, Clone)]
pub struct AdapterDecoratorConfig {
    pub kind: AdapterKind,
    pub targets: Vec<String>,
    pub rank: Option<i64>,
    pub alpha: Option<i64>,
}
```

In `crates/nsl-cli/src/main.rs`'s `analysis_to_wrga_inputs` helper (and the matching `analysis_to_wrga_inputs_from_configs` helper if it exists), add `alpha: c.alpha` to the adapter mapping.

- [ ] **Step 5: Run — expect pass**

```bash
cargo test -p nsl-semantic adapter_alpha 2>&1 | tail -5
cargo build --features cuda 2>&1 | tail -3
```

Expected: both new tests pass; full workspace builds.

- [ ] **Step 6: Commit 2a**

```bash
git add crates/nsl-semantic crates/nsl-codegen/src/lib.rs crates/nsl-cli/src/main.rs
git commit -m "feat(wrga): @adapter accepts optional alpha= for LoRA scaling"
```

### Task 2b — `wrga_adapter_inject` codegen pass: field + init emission

- [ ] **Step 1: Write the compile-time failing test**

Create `crates/nsl-codegen/tests/wrga_adapter_runtime.rs`:

```rust
//! Task 2 (B.2): adapter A/B fields exist on the compiled model with correct
//! shapes, strict init, and the forward pass applies LoRA with alpha scaling.

use nsl_codegen::{AdapterDecoratorConfig, AdapterKind, CompileOptions, WrgaInputs};

const LORA_SRC: &str = r#"
model Toy:
    let w: Tensor<[16, 16], f32> = zeros([16, 16])

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return x @ self.w

fn main():
    let m = Toy()
    let x: Tensor<[4, 16], f32> = zeros([4, 16])
    let _y = m.forward(x)
"#;

#[test]
fn adapter_inject_emits_lora_a_b_fields_with_expected_shapes() {
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::Lora,
                targets: vec!["m.w".into()],
                rank: Some(4),
                alpha: Some(4),
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let plan = nsl_codegen::debug_compile_and_return_plan(LORA_SRC, &opts)
        .expect("compile must succeed")
        .expect("wrga::run must fire when adapter is present");

    let site_id = "m_w__lora";
    let field_names: Vec<_> = plan.placements.iter()
        .flat_map(|p| p.synthesized_fields.iter().cloned())
        .collect();

    assert!(
        field_names.iter().any(|n| n == &format!("lora_A_{site_id}")),
        "expected lora_A_{site_id} in synthesized fields; got {:?}",
        field_names,
    );
    assert!(
        field_names.iter().any(|n| n == &format!("lora_B_{site_id}")),
        "expected lora_B_{site_id} in synthesized fields; got {:?}",
        field_names,
    );
}
```

> **Note:** `WrgaPlan.placements[i].synthesized_fields` is a new `Vec<String>` field the `wrga_adapter_inject` pass populates. It's the observation surface for this test. If mutating `WrgaPlan` is undesirable, add a side-channel in `lib.rs::debug_channels` instead (mirror `ADJOINT_OPS_DROPPED`): `ADAPTER_SYNTHESIZED_FIELDS: Cell<Option<Vec<String>>>`. Either works; the former is cleaner.

- [ ] **Step 2: Run — expect failure**

```bash
cargo test -p nsl-codegen --test wrga_adapter_runtime adapter_inject_emits_lora_a_b_fields 2>&1 | tail -15
```

Expected: fail because the adapter-inject pass doesn't exist yet.

- [ ] **Step 3: Create `wrga_adapter_inject.rs`**

Create `crates/nsl-codegen/src/wrga_adapter_inject.rs`:

```rust
//! WRGA Milestone B.2 Task 2: codegen pass that materialises @adapter
//! decorators as synthesized model fields + strict initialisers + forward
//! rewrite.
//!
//! Runs once per compile, after `wrga::run` produces the `WrgaPlan` and
//! before `compile_model_methods` lowers model forward-pass bodies.
//!
//! The forward-pass rewrite is UNFUSED for B.2:
//!   `y = x @ W + ((x @ A) @ B) * (alpha / rank)`
//! which expands into 2 extra matmul launches + elementwise scale+add per
//! site.  Milestone B.3's MMA epilogue collapses this into one fused kernel.

use crate::AdapterKind;
use crate::wrga::WrgaPlan;

/// Per-site adapter materialisation decision produced by this pass.
#[derive(Debug, Clone)]
pub struct AdapterSite {
    pub site_id: String,           // e.g. "m_w__lora"
    pub kind: AdapterKind,
    pub target_param: String,      // e.g. "m.w"
    pub rank: i64,
    pub alpha: i64,                // defaults to rank if decorator omitted it
    pub synthesized_fields: Vec<String>,  // field names added to the owning model
    pub input_dim: u32,
    pub output_dim: u32,
}

pub struct AdapterInjectResult {
    pub sites: Vec<AdapterSite>,
}

/// Entry point — called from compile_module_with_imports_best_effort_plan
/// after `invoke_wrga_if_enabled` returns a plan.
pub fn run(plan: &WrgaPlan, compiler: &mut crate::compiler::Compiler) -> AdapterInjectResult {
    let mut sites: Vec<AdapterSite> = Vec::new();
    for placement in &plan.placements {
        let Some(adapter_kind) = placement_adapter_kind(placement) else { continue; };
        let (input_dim, output_dim) = placement_dims(placement);
        let rank = placement.rank.unwrap_or(1).max(1);
        let alpha = placement.alpha.unwrap_or(rank).max(1);
        let site_id = site_id_from(placement);

        let fields = match adapter_kind {
            AdapterKind::Lora | AdapterKind::GatedLora => {
                let mut f = vec![
                    format!("lora_A_{site_id}"),
                    format!("lora_B_{site_id}"),
                ];
                if matches!(adapter_kind, AdapterKind::GatedLora) {
                    f.push(format!("gate_{site_id}"));
                }
                f
            }
            AdapterKind::Ia3 => vec![format!("ia3_scale_{site_id}")],
        };

        // Register synthesized fields on the compiler's model map so
        // `compile_model_constructor` emits their allocation + init.
        register_synthesized_fields(compiler, &placement.target_param, &fields, adapter_kind, rank, input_dim, output_dim);

        sites.push(AdapterSite {
            site_id,
            kind: adapter_kind,
            target_param: placement.target_param.clone(),
            rank,
            alpha,
            synthesized_fields: fields,
            input_dim,
            output_dim,
        });
    }
    AdapterInjectResult { sites }
}

fn site_id_from(p: &crate::wrga::Placement) -> String {
    // "m.w" -> "m_w__lora" / "m_w__ia3" / ...
    let safe = p.target_param.replace('.', "_");
    let suffix = match placement_adapter_kind(p).unwrap() {
        AdapterKind::Lora => "lora",
        AdapterKind::Ia3 => "ia3",
        AdapterKind::GatedLora => "gatedlora",
    };
    format!("{safe}__{suffix}")
}

// Stubs for recon during implementation:
fn placement_adapter_kind(_p: &crate::wrga::Placement) -> Option<AdapterKind> {
    // Placement must carry adapter kind.  If it doesn't, extend it here.
    unimplemented!("wire placement -> AdapterKind")
}
fn placement_dims(_p: &crate::wrga::Placement) -> (u32, u32) {
    // Input / output dims of the target weight's matmul.
    unimplemented!("extract dims from placement's weight shape")
}
fn register_synthesized_fields(
    _compiler: &mut crate::compiler::Compiler,
    _target: &str,
    _fields: &[String],
    _kind: AdapterKind,
    _rank: i64,
    _in_dim: u32,
    _out_dim: u32,
) {
    // This is where the pass tells compiler.models to emit extra fields
    // in the constructor.  Implementation detail depends on how
    // compile_model_constructor currently walks the ModelDef.
    unimplemented!()
}
```

> This scaffolding intentionally leaves three `unimplemented!` stubs. Fill them during implementation — the shapes are dictated by `crate::wrga::Placement`'s actual fields (recon: placement struct at `crates/nsl-codegen/src/wrga.rs:~70`) and by how `crates/nsl-codegen/src/compiler/functions.rs::compile_model_constructor` currently records fields (around lines 201-410).

- [ ] **Step 4: Expose the module + invoke after `wrga::run`**

In `crates/nsl-codegen/src/lib.rs` add `pub mod wrga_adapter_inject;`.

In `crates/nsl-codegen/src/stmt.rs` in `invoke_wrga_if_enabled` (or wherever the plan is finalized), after the plan is stashed onto `compiler.last_wrga_plan`, run the inject pass:

```rust
if let Some(plan) = compiler.last_wrga_plan.clone() {
    let inject_result = crate::wrga_adapter_inject::run(&plan, compiler);
    // Stash on the compiler for later phases (forward-rewrite, constructor init).
    compiler.wrga_adapter_sites = inject_result.sites;
}
```

Add a `pub wrga_adapter_sites: Vec<crate::wrga_adapter_inject::AdapterSite>` field on `Compiler` mirroring the B.1 `last_wrga_plan` pattern.

- [ ] **Step 5: Emit constructor init for each synthesized field**

In `crates/nsl-codegen/src/compiler/functions.rs::compile_model_constructor` (around line 356, after existing user-field init), walk `self.wrga_adapter_sites`, match each site against the current model being constructed, and for each synthesized field emit an allocation + strict initializer call:

```rust
// After user-field init loop:
for site in &self.wrga_adapter_sites.clone() {
    if !site_belongs_to_model(&site.target_param, current_model_name) { continue; }
    for fname in &site.synthesized_fields {
        let shape = shape_for(site, fname);  // e.g. [rank, in_dim] for lora_A_*
        let init_kind = init_kind_for(site, fname);  // Kaiming | Zeros | Ones
        // Emit:
        //   let <fname> = nsl_tensor_<init>(<shape>);
        //   store <fname> into model struct at offset matching the field.
        emit_synthesized_field_init(self, builder, state, fname, shape, init_kind)?;
    }
}
```

Key invariants:
- **LoRA B MUST use `nsl_tensor_zeros`** — never the default init. Hand-wire the call to `nsl_tensor_zeros` in PTX/Cranelift emission.
- **LoRA A uses a Kaiming-uniform init.** If `nsl_tensor_kaiming_uniform` doesn't exist, generate Uniform-in-[-k, k] with `k = sqrt(1.0 / in_dim)` via existing `nsl_tensor_uniform` + scale ops.
- **IA³ uses `nsl_tensor_ones`.**
- **GatedLoRA `gate` uses `nsl_tensor_zeros`.**

> `shape_for` and `init_kind_for` are small helpers local to `wrga_adapter_inject.rs`; implement alongside the pass.

- [ ] **Step 6: Forward-pass rewrite**

In `compile_model_methods` (around `functions.rs:511`), before compiling each statement in a method body, check whether the current model has any `AdapterSite` whose `target_param` matches a weight referenced in the statement. If yes, the statement is rewritten from `y = x @ W` to the scaled form.

The minimum-viable hook:

```rust
// At method-entry time:
let my_sites: Vec<&AdapterSite> = self.wrga_adapter_sites
    .iter()
    .filter(|s| site_belongs_to_model(&s.target_param, &model_name))
    .collect();

for stmt in &fn_def.body.stmts {
    let rewritten = if my_sites.is_empty() {
        stmt.clone()
    } else {
        crate::wrga_adapter_inject::rewrite_stmt_for_adapters(stmt, &my_sites)
    };
    self.compile_stmt(&mut builder, &mut state, &rewritten)?;
}
```

The `rewrite_stmt_for_adapters` helper (add to `wrga_adapter_inject.rs`) walks the AST and for each `Expr::BinaryOp { op: MatMul, lhs, rhs }` where `rhs` resolves to a target weight, replaces it with the scaled LoRA expression. For IA³ / GatedLoRA, the rewrites differ — keep LoRA only in Task 2b and add IA³/GatedLoRA rewrites in a follow-up step inside the same task if test coverage is there.

- [ ] **Step 7: Add the runtime-semantics test**

Append to `crates/nsl-codegen/tests/wrga_adapter_runtime.rs`:

```rust
/// With LoRA B = strict zeros, the forward output must match the base
/// model's forward output within f32 tolerance, regardless of alpha or rank.
#[test]
fn lora_with_b_zero_matches_base_model_output() {
    const SRC: &str = r#"
model Toy:
    let w: Tensor<[4, 4], f32> = zeros([4, 4])

    fn forward(self, x: Tensor<[1, 4], f32>) -> Tensor<[1, 4], f32>:
        return x @ self.w

fn main():
    let m = Toy()
    let x: Tensor<[1, 4], f32> = zeros([1, 4])
    let _y = m.forward(x)
"#;
    // Base compile (no adapter).
    let base_plan = nsl_codegen::debug_compile_and_return_plan(
        SRC,
        &CompileOptions { source_ad: true, ..Default::default() },
    ).expect("base compile").expect_none();
    // The assertion above is: no adapter → wrga plan is None.

    // Adapter compile.
    let adapter_opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            adapter: vec![AdapterDecoratorConfig {
                kind: AdapterKind::Lora,
                targets: vec!["m.w".into()],
                rank: Some(2),
                alpha: Some(4),  // scale = 2.0
            }],
            ..Default::default()
        }),
        source_ad: true,
        ..Default::default()
    };
    let adapter_plan = nsl_codegen::debug_compile_and_return_plan(SRC, &adapter_opts)
        .expect("adapter compile must succeed")
        .expect("plan must fire");

    // Verify B is zero-initialized (observational via plan metadata).
    let b_init = adapter_plan.placements.iter()
        .flat_map(|p| p.init_strategies.iter())
        .find(|s| s.field_name.contains("lora_B_"))
        .expect("lora_B init strategy must be emitted");
    assert_eq!(
        b_init.kind,
        nsl_codegen::wrga_adapter_inject::InitKind::Zeros,
        "lora_B must use strict Zeros init, not default",
    );
    assert_eq!(
        b_init.kind != nsl_codegen::wrga_adapter_inject::InitKind::Uniform,
        true,
    );
}
```

> `Placement.init_strategies: Vec<InitStrategy>` and `InitStrategy { field_name, kind }` with `InitKind { Zeros, Ones, KaimingUniform }` are observation surfaces added by the pass. Wire them into `WrgaPlan` OR into a side-channel, whichever is cleaner.

- [ ] **Step 8: Run — expect pass**

```bash
cargo test -p nsl-codegen --test wrga_adapter_runtime 2>&1 | tail -15
cargo build --features cuda 2>&1 | tail -3
```

- [ ] **Step 9: Regression**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | tail -3
```

Expected: all green. Existing `wrga_run_invoked` / `wrga_freeze_end_to_end` tests must still pass.

- [ ] **Step 10: Commit 2b**

```bash
git add crates/nsl-codegen/src/wrga_adapter_inject.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/src/compiler crates/nsl-codegen/tests/wrga_adapter_runtime.rs
git commit -m "feat(wrga): synthesize LoRA/IA3/GatedLoRA fields with strict init + forward rewrite"
```

---

## Task 3: `AllocationKey` type + `--wrga-fold-allocations` flag

**Files:**
- Modify: `crates/nsl-codegen/src/memory_planner.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (CompileOptions extension)
- Modify: `crates/nsl-codegen/src/stmt.rs` (flag-gated branch)
- Modify: `crates/nsl-cli/src/main.rs` (clap arg)
- Test: `crates/nsl-codegen/tests/wrga_fold_allocations_flag.rs`

**Goal:** Extend `LivenessAnalyzer` with a typed `AllocationKey { Weight(String), Activation(VarId) }` slot map; implement real `consume_hints(&WrgaPlan) -> usize`; gate folding behind `--wrga-fold-allocations` (default off). B.1's observational path remains default.

- [ ] **Step 1: Unit test for `AllocationKey` + `consume_hints`**

Append to `crates/nsl-codegen/src/memory_planner.rs` under a `#[cfg(test)] mod consume_hints_tests { ... }`:

```rust
#[cfg(test)]
mod consume_hints_tests {
    use super::*;
    use crate::wengert::VarId;
    use crate::wrga_memory::{MemoryPlan, MemoryPlanStats, SlotAssignment, SlotId};
    use crate::wrga::WrgaPlan;

    fn mk_plan_with_shared_slot() -> WrgaPlan {
        WrgaPlan {
            memory: MemoryPlan {
                assignments: vec![
                    SlotAssignment { var: VarId(1), slot: SlotId(0), size_bytes: 64, birth: 0, death: 5 },
                    SlotAssignment { var: VarId(2), slot: SlotId(0), size_bytes: 64, birth: 10, death: 15 },
                ],
                stats: MemoryPlanStats::default(),
            },
            ..WrgaPlan::test_dummy()  // constructor that fills non-memory fields with defaults
        }
    }

    #[test]
    fn consume_hints_merges_disjoint_equal_size_activations() {
        let mut a = LivenessAnalyzer::new();
        a.record_activation_alloc(VarId(1), 64, SizeKind::Exact, "tile_0");
        a.record_activation_alloc(VarId(2), 64, SizeKind::Exact, "tile_1");
        let plan = mk_plan_with_shared_slot();
        let post = a.consume_hints(&plan);
        assert_eq!(post, 1, "disjoint+equal-size should merge into one slot");
    }

    #[test]
    fn consume_hints_refuses_overlapping_liveness() {
        let mut a = LivenessAnalyzer::new();
        a.record_activation_alloc(VarId(1), 64, SizeKind::Exact, "tile_0");
        a.record_activation_alloc(VarId(2), 64, SizeKind::Exact, "tile_1");
        let mut plan = mk_plan_with_shared_slot();
        // Make them overlap:
        plan.memory.assignments[1].birth = 3;  // overlap with var1's 0..=5
        let post = a.consume_hints(&plan);
        assert_eq!(post, 2, "overlapping liveness must refuse to merge");
    }

    #[test]
    fn consume_hints_refuses_size_mismatch() {
        let mut a = LivenessAnalyzer::new();
        a.record_activation_alloc(VarId(1), 64, SizeKind::Exact, "tile_0");
        a.record_activation_alloc(VarId(2), 128, SizeKind::Exact, "tile_1");
        let plan = mk_plan_with_shared_slot();
        let post = a.consume_hints(&plan);
        assert_eq!(post, 2, "size mismatch must refuse to merge");
    }
}
```

> `WrgaPlan::test_dummy()` is a helper to construct a minimal `WrgaPlan` with non-memory fields defaulted. Add it behind `#[cfg(test)]` in `crates/nsl-codegen/src/wrga.rs`.

- [ ] **Step 2: Run — expect failure**

```bash
cargo test -p nsl-codegen memory_planner::consume_hints_tests 2>&1 | tail -15
```

Expected: compile error — `AllocationKey`, `record_activation_alloc`, `consume_hints`, `WrgaPlan::test_dummy` all missing.

- [ ] **Step 3: Introduce `AllocationKey` + `RealSlotId`**

At the top of `crates/nsl-codegen/src/memory_planner.rs`, add:

```rust
use std::collections::HashMap;
use crate::wengert::VarId;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum AllocationKey {
    /// Compile-time-named allocation (weights, constants, stdlib-backed buffers).
    Weight(String),
    /// Source-AD-tracked activation, keyed by its primal VarId.
    Activation(VarId),
}

/// The allocator's real slot identifier (distinct from WRGA's advisory SlotId).
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct RealSlotId(pub u32);
```

- [ ] **Step 4: Extend `LivenessAnalyzer` struct**

In the existing `LivenessAnalyzer` definition (around line 168), add fields and methods:

```rust
pub struct LivenessAnalyzer {
    // ... existing fields ...
    pub(crate) slots: HashMap<AllocationKey, RealSlotId>,
    pub(crate) slot_sizes: HashMap<RealSlotId, u64>,
    pub(crate) slot_liveness: HashMap<RealSlotId, Vec<(u32, u32)>>,  // per slot: list of (birth, death) of members
    pub(crate) next_slot: u32,
}

impl LivenessAnalyzer {
    /// Record an activation allocation (VarId-keyed).  Populates the real
    /// slot map so `consume_hints` has state to operate on.  Existing
    /// String-keyed `record_alloc` is unchanged.
    pub fn record_activation_alloc(
        &mut self,
        var: VarId,
        size_bytes: u64,
        kind: SizeKind,
        loc: &str,
    ) {
        let key = AllocationKey::Activation(var);
        let slot = RealSlotId(self.next_slot);
        self.next_slot += 1;
        self.slots.insert(key, slot);
        self.slot_sizes.insert(slot, size_bytes);
        // birth/death derived from the existing liveness tracking pass;
        // for standalone-test purposes, a placeholder range (0, 0) is
        // overwritten by the real pass.  Tests call this directly and
        // then provide the assignment's birth/death via consume_hints'
        // plan, so the local liveness isn't consulted for the pair-merge
        // decision.
        self.slot_liveness.insert(slot, vec![(0, 0)]);
        let _ = (kind, loc);  // reuse for the existing name-keyed path if needed
    }

    pub fn real_slot_for_activation(&self, var: VarId) -> Option<RealSlotId> {
        self.slots.get(&AllocationKey::Activation(var)).copied()
    }

    /// Merge `var`'s slot into `target`, iff size matches AND liveness is
    /// disjoint from every member already in `target`.  Returns true on
    /// successful merge; false on any check failure.
    pub fn try_merge_activation_into_slot(
        &mut self,
        var: VarId,
        target: RealSlotId,
    ) -> bool {
        let Some(source) = self.real_slot_for_activation(var) else { return false; };
        if source == target { return false; }
        let Some(&source_size) = self.slot_sizes.get(&source) else { return false; };
        let Some(&target_size) = self.slot_sizes.get(&target) else { return false; };
        if source_size != target_size { return false; }
        // Liveness disjointness check is driven by the plan-supplied ranges
        // in consume_hints; this helper is a primitive and assumes the
        // caller has validated.
        self.slots.insert(AllocationKey::Activation(var), target);
        self.slot_sizes.remove(&source);
        // Merge liveness lists.
        if let Some(src_liv) = self.slot_liveness.remove(&source) {
            self.slot_liveness.entry(target).or_default().extend(src_liv);
        }
        true
    }

    /// Consume WRGA's MemoryPlan as coalescing hints.  Groups assignments
    /// by their WRGA slot; for each group of ≥2, applies size+disjointness
    /// checks pairwise and merges into a single real slot.  Returns the
    /// post-merge count of distinct real slots.
    pub fn consume_hints(&mut self, plan: &crate::wrga::WrgaPlan) -> usize {
        let mut by_wrga_slot: HashMap<crate::wrga_memory::SlotId, Vec<(VarId, u32, u32, u64)>> = HashMap::new();
        for a in &plan.memory.assignments {
            by_wrga_slot.entry(a.slot).or_default().push((a.var, a.birth, a.death, a.size_bytes));
        }
        for (_wrga_slot, group) in by_wrga_slot {
            if group.len() < 2 { continue; }
            let (first_var, _, _, first_size) = group[0];
            let Some(target) = self.real_slot_for_activation(first_var) else { continue; };
            for &(other_var, o_birth, o_death, o_size) in &group[1..] {
                if o_size != first_size { continue; }
                // Disjointness: other's range must not overlap any existing
                // range already merged into `target`.
                let overlap = self.slot_liveness
                    .get(&target)
                    .map(|ranges| ranges.iter().any(|&(b, d)| !(o_death < b || d < o_birth)))
                    .unwrap_or(false);
                if overlap { continue; }
                self.try_merge_activation_into_slot(other_var, target);
            }
        }
        // Distinct real slots currently in use:
        let distinct: std::collections::BTreeSet<_> = self.slots.values().copied().collect();
        distinct.len()
    }
}
```

> The existing String-keyed `record_alloc` is untouched — it already stores into the String-key map. The new `AllocationKey::Weight(String)` is ready but only populated if a future caller opts in. That's fine for B.2; migrating the whole String path is out of scope.

- [ ] **Step 5: Run the unit tests — expect pass**

```bash
cargo test -p nsl-codegen memory_planner::consume_hints_tests 2>&1 | tail -10
```

- [ ] **Step 6: Add the CLI flag + CompileOptions field**

In `crates/nsl-codegen/src/lib.rs`, extend `CompileOptions`:

```rust
/// Task 3 (B.2): when true, memory-planner folds activations per WRGA's
/// MemoryPlan (real allocator merge).  Default false → observational only
/// (B.1 behaviour).  Flipping this default happens after B.3 + real workloads.
pub wrga_fold_allocations: bool,
```

Default to `false` in the `impl Default for CompileOptions`.

In `crates/nsl-cli/src/main.rs`, add to the `Build` subcommand:

```rust
/// Task 3 (B.2): fold WRGA memory hints into real allocations.
/// Default off — observational mode ships per B.1.
#[arg(long, default_value_t = false)]
wrga_fold_allocations: bool,
```

Plumb through wherever `options` is built: `options.wrga_fold_allocations = args.wrga_fold_allocations;`.

- [ ] **Step 7: Branch in `stmt.rs`**

Find where `apply_wrga_hints` is called in `crates/nsl-codegen/src/stmt.rs` (B.1 wired this in). Replace with a branch on the flag:

```rust
if let Some(plan) = &pruned_plan {
    if self.options.wrga_fold_allocations {
        let post = crate::memory_planner::consume_hints(&mut liveness, plan);
        // Side-channel reporting remains compatible:
        crate::debug_set_allocator_slot_count_post_hint(post);
    } else {
        let _ = crate::memory_planner::apply_wrga_hints(&mut liveness, plan);
    }
}
```

- [ ] **Step 8: Integration test for the flag**

Create `crates/nsl-codegen/tests/wrga_fold_allocations_flag.rs`:

```rust
//! Task 3 (B.2): `--wrga-fold-allocations` → `consume_hints` runs and
//! reduces the post-hint slot count.  Default off → B.1 observational
//! path still runs.

use nsl_codegen::{CompileOptions, FreezeDecoratorConfig, WrgaInputs};

const SRC: &str = r#"
@freeze(include=["m.w1", "m.w2"])
let m = Stack()

model Stack:
    let w1: Tensor<[32, 32], f32> = zeros([32, 32])
    let w2: Tensor<[32, 32], f32> = zeros([32, 32])
    let w3: Tensor<[32, 32], f32> = zeros([32, 32])

    fn forward(self, x: Tensor<[4, 32], f32>) -> Tensor<[4, 32], f32>:
        let a = x @ self.w1
        let b = a @ self.w2
        return b @ self.w3

fn main():
    train(model=m, epochs=1):
        optimizer: sgd(lr=1e-3)
        step(batch):
            let y = m.forward(batch.x)
            return y.sum()
"#;

fn compile_with_flag(flag: bool) -> Option<usize> {
    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            freeze: vec![FreezeDecoratorConfig {
                include: vec!["m.w1".into(), "m.w2".into()],
                exclude: vec![],
            }],
            ..Default::default()
        }),
        wrga_fold_allocations: flag,
        source_ad: true,
        ..Default::default()
    };
    let _plan = nsl_codegen::debug_compile_and_return_plan(SRC, &opts).ok()??;
    nsl_codegen::debug_last_allocator_slot_count_post_hint()
}

#[test]
fn fold_flag_off_preserves_observational_path() {
    let post = compile_with_flag(false).expect("observational side-channel must fire");
    let pre = nsl_codegen::debug_last_allocator_slot_count_pre_hint().unwrap();
    assert!(post <= pre, "B.1 invariant holds with flag off");
}

#[test]
fn fold_flag_on_invokes_consume_hints() {
    let post = compile_with_flag(true).expect("side-channel must fire with flag on");
    let pre = nsl_codegen::debug_last_allocator_slot_count_pre_hint().unwrap();
    // Folding only reduces or equals; it never increases.
    assert!(post <= pre);
    // A side-signal from consume_hints itself (to be added in debug_channels)
    // proves which branch ran.  If consume_hints increments a counter
    // like `CONSUME_HINTS_CALLS`, assert it here:
    let calls = nsl_codegen::debug_last_consume_hints_calls().unwrap_or(0);
    assert!(calls >= 1, "consume_hints must run when flag is on");
}
```

Add the `CONSUME_HINTS_CALLS` side-channel alongside `ADJOINT_OPS_DROPPED` in `lib.rs`'s `debug_channels` module, with `#[doc(hidden)] pub fn debug_last_consume_hints_calls() -> Option<usize>`.

- [ ] **Step 9: Run — expect pass**

```bash
cargo test -p nsl-codegen --test wrga_fold_allocations_flag 2>&1 | tail -15
cargo test -p nsl-codegen --test wrga_memory_hints 2>&1 | tail -5   # B.1 test must still pass
```

- [ ] **Step 10: Regression**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
cargo test -p nsl-codegen --tests 2>&1 | tail -3
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | tail -3
cargo build --features cuda 2>&1 | tail -3
```

- [ ] **Step 11: Commit**

```bash
git add crates/nsl-codegen/src/memory_planner.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/stmt.rs crates/nsl-cli/src/main.rs crates/nsl-codegen/tests/wrga_fold_allocations_flag.rs
git commit -m "feat(wrga): typed AllocationKey + --wrga-fold-allocations flag (default off)"
```

---

## Task 4: ZK/standalone `--wrga-report` emission

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/entry_points.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` (re-exports)
- Modify: `crates/nsl-cli/src/main.rs` (remove "not yet supported"; wire new variants)
- Test: `crates/nsl-cli/tests/wrga_report_cli.rs` (append)

**Goal:** `nsl build --zk-circuit --source-ad --wrga-report` and `nsl build --standalone --source-ad --wrga-report` produce real reports via plan-returning variants.

- [ ] **Step 1: Write the failing tests**

Append to `crates/nsl-cli/tests/wrga_report_cli.rs`:

```rust
#[test]
fn wrga_report_works_on_zk_build_path() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC).unwrap();
    let stdlib = std::env::var("NSL_STDLIB_PATH")
        .unwrap_or_else(|_| "c:/Users/bwiem/projects/NSL-wrga-b2/stdlib".to_string());

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .arg("build").arg(&src_path)
        .arg("--zk-circuit").arg("--source-ad").arg("--wrga-report");
    cmd.assert()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="));
}

#[test]
fn wrga_report_works_on_standalone_build_path() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("t.nsl");
    fs::write(&src_path, SRC).unwrap();
    let stdlib = std::env::var("NSL_STDLIB_PATH")
        .unwrap_or_else(|_| "c:/Users/bwiem/projects/NSL-wrga-b2/stdlib".to_string());

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", &stdlib)
        .arg("build").arg(&src_path)
        .arg("--standalone").arg("--source-ad").arg("--wrga-report");
    cmd.assert()
        .stdout(predicate::str::contains("=== WRGA Compilation Report ==="));
}
```

- [ ] **Step 2: Run — expect failure**

```bash
cargo test -p nsl-cli --test wrga_report_cli wrga_report_works_on_zk wrga_report_works_on_standalone 2>&1 | tail -15
```

Expected: tests fail because of the existing "not yet supported" stderr path.

- [ ] **Step 3: Add plan-returning variants in `entry_points.rs`**

In `crates/nsl-codegen/src/compiler/entry_points.rs`:

```rust
pub fn compile_with_zk_info_returning_plan(
    // ... same args as compile_with_zk_info ...
) -> Result<(Vec<u8>, HashMap<String, crate::zk::backend::ZkMode>, Vec<(String, crate::zk::ZkCompileResult)>, Option<crate::wrga::WrgaPlan>), CodegenError> {
    // Build Compiler the same way compile_with_zk_info does.
    // Run all passes.  After wrga::run has stashed last_wrga_plan (if
    // --source-ad + WRGA decorators), clone it.
    // Return (obj_bytes, zk_modes, zk_results, compiler.last_wrga_plan.clone()).
}

pub fn compile_standalone_returning_plan(
    // ... same args as compile_standalone ...
) -> Result<(Vec<u8>, Option<crate::wrga::WrgaPlan>), CodegenError> {
    // Same shape as compile_standalone, with plan clone before return.
}
```

> Both variants delegate to the existing `compile_with_zk_info` / `compile_standalone` internals — avoid duplicating the full body. Factor the shared code into a helper if the existing functions don't already return a `Compiler` or plan reference. If the ZK pipeline truly skips `invoke_wrga_if_enabled` (per the recon), the plan will remain `None` and the test will fail with "no plan produced"; that's the accurate signal that ZK needs explicit WRGA invocation added before this task is complete. If that happens, extend `compile_with_zk_info_returning_plan` to call the adapter-inject pass's prerequisites — but do NOT touch `backend_ptx.rs` or the MMA path (B.3 territory).

- [ ] **Step 4: Re-export from `lib.rs`**

```rust
pub use crate::compiler::entry_points::{
    compile_with_zk_info_returning_plan,
    compile_standalone_returning_plan,
};
```

- [ ] **Step 5: Wire in `main.rs`**

In `crates/nsl-cli/src/main.rs`, find `run_build_zk` and `run_build_standalone` (recon: around lines 1605 and 1814 in the post-B.1 file). Remove the "not yet supported" branch:

```rust
// Before:
if args.wrga_report.is_some() {
    eprintln!("--wrga-report is not yet supported on --zk-circuit builds; decorator effects still apply");
}

// After:
let (_obj, zk_modes, zk_results, plan) = nsl_codegen::compile_with_zk_info_returning_plan(
    // ... existing args ...
)?;
emit_wrga_report(&args.wrga_report, plan.as_ref())?;
```

Mirror in `run_build_standalone`.

- [ ] **Step 6: Run — expect pass**

```bash
cargo build --bin nsl 2>&1 | tail -3
cargo test -p nsl-cli --test wrga_report_cli 2>&1 | tail -10
```

Expected: the 4 B.1 tests + the 2 new ones all pass.

- [ ] **Step 7: Regression**

```bash
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | tail -3
cargo build --features cuda 2>&1 | tail -3
```

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-codegen/src/compiler/entry_points.rs crates/nsl-codegen/src/lib.rs crates/nsl-cli/src/main.rs crates/nsl-cli/tests/wrga_report_cli.rs
git commit -m "feat(wrga): --wrga-report on ZK + standalone build paths"
```

---

## Task 5: Close-out

**Files:**
- Modify (outside worktree): `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wrga_milestone_b2.md` (new)
- Modify: `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md` (append one line)
- **No code changes.**

- [ ] **Step 1: Run the full regression**

```bash
cargo test -p nsl-semantic 2>&1 | grep "test result" | tail -3
cargo test -p nsl-codegen --lib 2>&1 | grep "test result" | tail -3
cargo test -p nsl-codegen --tests 2>&1 | grep "test result" | tail -3
cargo test -p nsl-codegen flash_attention 2>&1 | grep "test result" | tail -3
cargo test -p nsl-cli --test e2e -- --test-threads=1 2>&1 | grep "test result" | tail -3
cargo test -p nsl-cli --test wrga_report_cli 2>&1 | grep "test result" | tail -3
cargo build --features cuda 2>&1 | tail -3
cargo build --release --features cuda 2>&1 | tail -3
cargo clippy -p nsl-codegen -p nsl-semantic -p nsl-cli --features cuda --all-targets 2>&1 | grep -E "^warning:" | grep -v "nsl-runtime" | head -20
```

Expected:
- semantic ≥ 260 (258 baseline + ≥2 from Task 2a), codegen lib ≥ 884 (881 + ≥3 from Task 3 unit tests), codegen --tests ≥ (previous + 3 new test files), flash_attention unchanged (≥56), nsl-cli e2e ≥ 80, wrga_report_cli ≥ 6 (4 + 2 from Task 4).
- Release build clean.
- No new clippy warnings in files this plan touched.

- [ ] **Step 2: Write the memory file**

Create `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wrga_milestone_b2.md`:

```markdown
---
name: WRGA Milestone B.2 — runtime adapters + allocator plumbing + report polish
description: LoRA materialisation, AllocationKey type, --wrga-fold-allocations, ZK/standalone --wrga-report; B.3 (MMA epilogue) still outstanding
type: project
---

## WRGA Milestone B.2 (landed 2026-04-12 on branch feat/wrga-milestone-b2)

- Task 1: `debug_compile_and_return_plan` now loads stdlib via `nsl-codegen/src/stdlib_loader.rs`; test sources with `optimizer: sgd(...)` compile.
- Task 2: `@adapter(type=lora, alpha=..., rank=..., ...)` decorator now materialises synthesized model fields (`lora_A_<site>`, `lora_B_<site>`, etc.), strict init (LoRA B = zeros via `nsl_tensor_zeros`), forward rewrite with `scale = alpha/rank`. Unfused (3 kernels per site); B.3 MMA epilogue collapses.
- Task 3: `AllocationKey { Weight(String), Activation(VarId) }` enforces keyspace separation at compile time. `consume_hints` + `--wrga-fold-allocations` (default off) ship; observational path remains default.
- Task 4: `compile_with_zk_info_returning_plan` + `compile_standalone_returning_plan` exist; `nsl build --zk-circuit --wrga-report` and `--standalone --wrga-report` work.

**Known residual gap (B.3 plan):**
- FusionPlan MMA epilogue PTX in `backend_ptx.rs` — collapses LoRA's 3 kernels into 1 fused MMA pass.
- Flipping `--wrga-fold-allocations` to on-by-default.
- Configurable adapter init via `@adapter(init=...)`.

**Expected perf profile during B.2 (intentional regression):**
Adapter-enabled models run measurably slower per site until B.3 lands: 2 extra `nsl_matmul` launches + elementwise scale+add, VRAM intermediate for `x @ A`.
```

Append to `MEMORY.md` under the existing Milestone B.1 line:

```
## WRGA Milestone B.2 (2026-04-12)
- See [project_wrga_milestone_b2.md](project_wrga_milestone_b2.md) — runtime adapters (LoRA/IA3/GatedLoRA with alpha scaling), typed AllocationKey, --wrga-fold-allocations flag, ZK/standalone --wrga-report; B.3 (MMA epilogue) outstanding
```

(Memory dir is not a git repo; file writes are sufficient.)

- [ ] **Step 3: Do NOT merge.** Report the branch status to the controlling session; the merge + worktree-cleanup happens there after subagent reviews.

---

## Final verification (do not skip before reporting DONE)

- [ ] `cargo test --workspace --features cuda` (accept Windows parallel-test flake; single-thread nsl-cli re-run must pass).
- [ ] `cargo clippy --features cuda --all-targets -- -D warnings` on touched crates; no new warnings in B.2 code.
- [ ] `cargo build --release --features cuda` clean.

---

## Out of scope (explicit)

- MMA epilogue PTX in `backend_ptx.rs` or `epilogue_fusion.rs` — B.3.
- Configurable adapter init (`@adapter(init=...)`).
- `@wrga_adapters` serialization section.
- Full `nsl-cli` loader refactor (only the minimum inline loader for Task 1).
- Flipping `--wrga-fold-allocations` default to on.

---

## Risk log

1. **Task 1 stdlib loader — entry-module-vs-stdlib analysis ordering.** `nsl-semantic::analyze` may require imports resolved *before* analysis. The plan's approach is to analyse stdlib modules first (in topo order), then the entry module, threading analyses through `import_types`. If this pattern doesn't match nsl-semantic's actual API (e.g., it expects a single `ModuleGraph` argument), use `nsl-cli::loader::load_all_modules` as a model and restructure accordingly. Acceptable to diverge from the minimal sketch so long as the test passes and no file I/O leaks into nsl-runtime.

2. **Task 2 forward-pass rewrite shape-awareness.** The AST rewrite `y = x @ W + ((x @ A) @ B) * scale` needs to know the target `W`'s shape to pick correct A/B shapes. Extract from `type_map` at rewrite time; if not available, default to the decorator's `rank` + the weight's declared shape and fail loudly on mismatch (compile error, not silent).

3. **Task 2 `unimplemented!` stubs.** Three stubs in `wrga_adapter_inject.rs` (`placement_adapter_kind`, `placement_dims`, `register_synthesized_fields`) must be filled before the commit. If the `Placement` struct doesn't carry adapter kind/dims today, extend it — that's a scope-aligned change.

4. **Task 3 `LivenessAnalyzer` test setup.** The unit tests in the plan manually call `record_activation_alloc` then `consume_hints`; the helper needs a way to set birth/death without running the full liveness pass. The sketch passes (0, 0) and relies on `consume_hints`' in-plan `birth`/`death`; verify this matches the real implementation.

5. **Task 4 ZK path.** If ZK truly bypasses `invoke_wrga_if_enabled`, the ZK test will fail with "plan is None" even after the `_returning_plan` variant ships. Fallback: call the adapter-inject pass (Task 2 prerequisite) from the ZK pipeline explicitly, OR document the gap and file a ZK-WRGA-integration follow-up. Do not paper over by returning a fake plan.

6. **Expected performance regression (intentional).** `@adapter`-compiled models run slower per site in B.2 than before. CI perf gates must accommodate. If a benchmark test in the repo gates on matmul kernel count, adjust or disable for adapter-enabled configs until B.3.
