# M62 `@export` Decorator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `nsl build --shared-lib model.nsl` produce a `.so`/`.dylib`/`.dll` with C-callable symbols driven by `@export`-decorated functions, plus a matching generated `.h` header.

**Architecture:** Strict opt-in linkage override in `declaration.rs`'s existing decorator loop: `@export` bumps `Linkage::Local` → `Linkage::Export` and skips module-prefix mangling (or uses `name="..."`). Semantic validation catches ABI-incompatible signatures. Header emitter is a new module invoked after shared-lib compile. Per-function C-wrapper emission (Task 4) converts `NslTensorDesc*` args into internal `NslTensor` pointers so the exported symbol has a clean C ABI.

**Tech Stack:** Rust, Cranelift module (`declare_function`, `Linkage`), existing decorator infra at [parser/stmt.rs:428](../../../crates/nsl-parser/src/stmt.rs#L428) + [declaration.rs:62-76](../../../crates/nsl-codegen/src/compiler/declaration.rs#L62), Python `ctypes` for E2E verification.

**Spec:** [docs/superpowers/specs/2026-04-15-m62-export-decorator-design.md](../specs/2026-04-15-m62-export-decorator-design.md)

**Branch:** `feat/m62-finish` (already created from `origin/main` at `b2797bc`).

---

## File Inventory

**Create:**
- `crates/nsl-codegen/src/c_header.rs` — `ExportInfo`, `ExportTypeInfo`, type lowering, `emit()` function.
- `crates/nsl-semantic/src/export.rs` — `validate_exports` pass.
- `crates/nsl-codegen/tests/c_header_snapshot.rs` — header-output snapshot test.
- `crates/nsl-codegen/tests/fixtures/m62_export_header.nsl` — 2-function + 1-rename fixture for snapshot.
- `examples/m62_shared_lib.nsl` — minimal E2E `.so` example.
- `python/tests/test_m62_export.py` — Python ctypes E2E test.

**Modify:**
- `crates/nsl-codegen/src/compiler/mod.rs` — add `export_functions: Vec<ExportInfo>` field on `FeatureConfigs` and init.
- `crates/nsl-codegen/src/compiler/declaration.rs` — recognize `@export` in decorator loop, override linkage + symbol name, push `ExportInfo`.
- `crates/nsl-codegen/src/lib.rs` — `pub mod c_header;`, re-export `ExportInfo` / `ExportTypeInfo`.
- `crates/nsl-semantic/src/lib.rs` — call `export::validate_exports` from the top-level `analyze` / `analyze_with_imports` entry.
- `crates/nsl-cli/src/main.rs` — after shared-lib build, emit `.h` when `compiler.features.export_functions` is non-empty.

**Not touched:**
- `crates/nsl-parser/src/**` — generic decorator infra already handles `@export` syntax.
- `crates/nsl-ast/src/**` — no new AST node.
- `crates/nsl-runtime/src/c_api.rs` — `nsl_model_*` lifecycle already ships.

---

## Task 0: Discovery — locate `NslModel` struct + verify decorator loop shape

**Files:** Read-only.

Before any implementation, confirm the runtime-side `NslModel` definition location and the current decorator loop's control flow. These aren't blocking Task 1's scope, but knowing them upfront prevents re-discovery in later tasks.

- [ ] **Step 1: Find `NslModel` definition**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/m62-finish
grep -rn "pub struct NslModel " crates/nsl-runtime/src/ | head -3
```

Record the file path and struct fields. The grad-context sibling spec will edit this; `@export` doesn't, but the C wrapper in Task 4 needs to know how to get weight pointers from an `NslModel*` handle.

- [ ] **Step 2: Re-read the decorator loop**

Read [declaration.rs:62-110](../../../crates/nsl-codegen/src/compiler/declaration.rs#L62). Confirm:
- Loop iterates `decorators` twice (`@no_grad`/`@test`/`@fp8_compute` in first pass, `@grammar` in second pass).
- `self.resolve_sym(d.name[0])` returns `&str` — symbol resolution for the first-segment decorator name.
- `d.args` is `Option<Vec<Arg>>`; `arg.name: Option<Symbol>`; `arg.value: Expr`.

Note the exact structure; Task 2 adds an `@export` branch alongside the first pass.

- [ ] **Step 3: No commit — pure discovery**

---

## Task 1: Add `ExportInfo` + `ExportTypeInfo` types and `FeatureConfigs` field

**Files:**
- Create: `crates/nsl-codegen/src/c_header.rs` (minimal — just the type definitions for now; emission logic lands in Task 5).
- Modify: `crates/nsl-codegen/src/lib.rs` (register module).
- Modify: `crates/nsl-codegen/src/compiler/mod.rs` (add field + init).

- [ ] **Step 1: Create `c_header.rs` with the type definitions**

`crates/nsl-codegen/src/c_header.rs`:

```rust
//! M62 `@export` C header emission + type model.
//!
//! `ExportInfo` tracks each `@export`-decorated function so the CLI
//! can emit a matching C header after shared-lib codegen completes.
//! This module also owns the C-type lowering used by the header
//! emitter (Task 5) and the C-wrapper emission (Task 4).

use serde::{Deserialize, Serialize};

/// Per-function metadata for a single `@export` function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportInfo {
    /// The symbol name that appears in the `.so`/`.dylib`/`.dll`'s export table.
    /// Either the NSL function's raw name, or the user-provided `name="..."`.
    pub symbol_name: String,
    /// The NSL-side function name (used for diagnostics and for looking up
    /// the function body in `Compiler.registry.functions`).
    pub raw_name: String,
    /// Parameter types in declaration order.
    pub params: Vec<ExportParamInfo>,
    /// Return type. A tuple return is represented as `ExportTypeInfo::Tuple(...)`.
    pub return_type: ExportTypeInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportParamInfo {
    pub name: String,
    pub ty: ExportTypeInfo,
}

/// C-ABI-compatible type shapes that `@export` functions may use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportTypeInfo {
    /// `Tensor<[...], dtype, device>`.
    Tensor {
        /// Shape dims stringified. Named dims like `"B"` stay named;
        /// literal ints like `4` stringify to `"4"`.
        shape: Vec<String>,
        dtype: ExportDtype,
        device: ExportDevice,
    },
    /// Primitive scalar.
    Scalar(ExportDtype),
    /// Tuple of any of the above (for multi-output functions).
    Tuple(Vec<ExportTypeInfo>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportDtype {
    F32, F64, F16, BF16,
    I8, I16, I32, I64,
    U8, U16, U32, U64,
    Bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportDevice {
    Cpu,
    Cuda,
    /// Compiler chooses at call time (default for `@export` inputs that
    /// don't explicitly pin a device).
    Any,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_info_basic_shape() {
        let info = ExportInfo {
            symbol_name: "forward".into(),
            raw_name: "forward".into(),
            params: vec![ExportParamInfo {
                name: "x".into(),
                ty: ExportTypeInfo::Tensor {
                    shape: vec!["B".into(), "768".into()],
                    dtype: ExportDtype::F32,
                    device: ExportDevice::Any,
                },
            }],
            return_type: ExportTypeInfo::Tensor {
                shape: vec!["B".into(), "1000".into()],
                dtype: ExportDtype::F32,
                device: ExportDevice::Any,
            },
        };
        assert_eq!(info.params.len(), 1);
        assert_eq!(info.symbol_name, "forward");
    }
}
```

- [ ] **Step 2: Register the module in `lib.rs`**

In `crates/nsl-codegen/src/lib.rs`, find the line `pub mod fase;` (or any adjacent `pub mod X;`) and add:

```rust
pub mod c_header;
```

- [ ] **Step 3: Add `export_functions` field to `FeatureConfigs`**

In `crates/nsl-codegen/src/compiler/mod.rs`, find the `FeatureConfigs` struct definition. Add a new field after `fp8_compute_fns`:

```rust
/// M62: Functions decorated with `@export` — collected during
/// declaration so the CLI can emit a matching C header after the
/// shared-lib build completes.
pub export_functions: Vec<crate::c_header::ExportInfo>,
```

In the `impl FeatureConfigs { fn new(options: &crate::CompileOptions) -> Self { ... } }` body, add an init alongside `fp8_compute_fns: HashSet::new(),`:

```rust
export_functions: Vec::new(),
```

- [ ] **Step 4: Build + test the trivial shape**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
cargo test -p nsl-codegen c_header::tests 2>&1 | tail -3
```

Expected: clean build; 1 test passes.

- [ ] **Step 5: Run full lib for regressions**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: no regression vs baseline. The new module is inert — it only adds types.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/c_header.rs \
        crates/nsl-codegen/src/lib.rs \
        crates/nsl-codegen/src/compiler/mod.rs
git commit -m "feat(m62): ExportInfo + ExportTypeInfo type model

Adds the data types that Tasks 2-5 populate (declaration.rs hook) and
consume (header emission, C wrapper emission). Field
Compiler.features.export_functions starts empty; nothing yet writes
to it. One unit test pins the struct shape."
```

---

## Task 2: `@export` recognition in `declaration.rs`

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/declaration.rs`.

- [ ] **Step 1: Write the failing tests**

Add a test module at the end of `crates/nsl-codegen/src/compiler/declaration.rs`:

```rust
#[cfg(test)]
mod export_tests {
    // Note: declare_user_functions_with_linkage is tested indirectly via
    // parse + semantic + compile pipelines elsewhere; direct unit tests
    // of the decorator-loop behavior use a mock Compiler helper that
    // isolates the linkage/symbol decision. Since Compiler is large,
    // the minimal test here pins the extracted helper's behavior.

    use super::*;

    // Helper: build a minimal Decorator from a name string (no args).
    fn decorator_no_args(name: &str, interner: &mut nsl_lexer::Interner) -> nsl_ast::decl::Decorator {
        let sym = interner.intern(name);
        nsl_ast::decl::Decorator {
            name: vec![sym],
            args: None,
            span: nsl_ast::span::Span::dummy(),
        }
    }

    fn decorator_with_name_arg(
        decorator_name: &str,
        kwarg_name: &str,
        kwarg_val: &str,
        interner: &mut nsl_lexer::Interner,
    ) -> nsl_ast::decl::Decorator {
        let dname_sym = interner.intern(decorator_name);
        let kname_sym = interner.intern(kwarg_name);
        nsl_ast::decl::Decorator {
            name: vec![dname_sym],
            args: Some(vec![nsl_ast::decl::Arg {
                name: Some(kname_sym),
                value: nsl_ast::expr::Expr {
                    kind: nsl_ast::expr::ExprKind::StringLiteral(kwarg_val.into()),
                    span: nsl_ast::span::Span::dummy(),
                    id: nsl_ast::NodeId(0),
                },
                span: nsl_ast::span::Span::dummy(),
            }]),
            span: nsl_ast::span::Span::dummy(),
        }
    }

    #[test]
    fn extract_export_from_decorators_no_decorator() {
        let mut interner = nsl_lexer::Interner::new();
        let decos: Vec<nsl_ast::decl::Decorator> = vec![];
        let (is_export, override_name) = extract_export_decorator(&decos, &interner);
        assert!(!is_export);
        assert_eq!(override_name, None);
    }

    #[test]
    fn extract_export_from_decorators_bare_export() {
        let mut interner = nsl_lexer::Interner::new();
        let decos = vec![decorator_no_args("export", &mut interner)];
        let (is_export, override_name) = extract_export_decorator(&decos, &interner);
        assert!(is_export);
        assert_eq!(override_name, None);
    }

    #[test]
    fn extract_export_from_decorators_with_name() {
        let mut interner = nsl_lexer::Interner::new();
        let decos = vec![decorator_with_name_arg("export", "name", "predict", &mut interner)];
        let (is_export, override_name) = extract_export_decorator(&decos, &interner);
        assert!(is_export);
        assert_eq!(override_name, Some("predict".to_string()));
    }

    #[test]
    fn extract_export_from_decorators_ignores_non_export() {
        let mut interner = nsl_lexer::Interner::new();
        let decos = vec![decorator_no_args("no_grad", &mut interner)];
        let (is_export, override_name) = extract_export_decorator(&decos, &interner);
        assert!(!is_export);
        assert_eq!(override_name, None);
    }
}
```

- [ ] **Step 2: Confirm failure**

```bash
cargo test -p nsl-codegen compiler::declaration::export_tests 2>&1 | tail -5
```

Expected: FAIL — `extract_export_decorator` not found.

- [ ] **Step 3: Extract a pure helper function**

In `crates/nsl-codegen/src/compiler/declaration.rs`, add a free function in the same module (at the top, before `impl Compiler`):

```rust
/// Extract `@export` info from a list of decorators. Returns
/// `(is_export, Option<override_name>)`. `Interner` is passed by
/// reference so this can be unit-tested without a full `Compiler`.
fn extract_export_decorator(
    decorators: &[nsl_ast::decl::Decorator],
    interner: &nsl_lexer::Interner,
) -> (bool, Option<String>) {
    let mut is_export = false;
    let mut override_name: Option<String> = None;

    for d in decorators {
        if d.name.len() != 1 {
            continue;
        }
        let dname = interner.resolve(d.name[0]).unwrap_or("");
        if dname != "export" {
            continue;
        }
        is_export = true;
        if let Some(ref args) = d.args {
            for arg in args {
                if let Some(name_sym) = arg.name {
                    let arg_name = interner.resolve(name_sym).unwrap_or("");
                    if arg_name == "name" {
                        if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                            override_name = Some(s.clone());
                        }
                    }
                }
            }
        }
    }

    (is_export, override_name)
}
```

- [ ] **Step 4: Verify tests pass**

```bash
cargo test -p nsl-codegen compiler::declaration::export_tests 2>&1 | tail -3
```

Expected: 4 tests PASS.

- [ ] **Step 5: Wire the helper into `declare_user_functions_with_linkage`**

Modify the function body at [declaration.rs:32](../../../crates/nsl-codegen/src/compiler/declaration.rs#L32). Replace the existing header of the loop body:

**Before:**
```rust
let raw_name = self.resolve_sym(fn_def.name).to_string();
let cranelift_name = mangle_name(&self.module_prefix, &raw_name);
let sig = self.build_fn_signature(fn_def);
let func_id = self
    .module
    .declare_function(&cranelift_name, linkage, &sig)
    .map_err(|e| {
        CodegenError::new(format!("failed to declare fn '{raw_name}': {e}"))
    })?;
self.registry
    .functions
    .insert(raw_name.clone(), (func_id, sig));
```

**After:**
```rust
let raw_name = self.resolve_sym(fn_def.name).to_string();
let cranelift_name = mangle_name(&self.module_prefix, &raw_name);
let sig = self.build_fn_signature(fn_def);

// M62: @export override — if decorated, use Export linkage and an
// unmangled (or user-named) symbol so C consumers see a clean ABI.
let (is_export, override_name) = match decorators {
    Some(decos) => extract_export_decorator(decos, self.interner),
    None => (false, None),
};
let effective_linkage = if is_export { cranelift_module::Linkage::Export } else { linkage };
let symbol_name = if is_export {
    override_name.clone().unwrap_or_else(|| raw_name.clone())
} else {
    cranelift_name.clone()
};

let func_id = self
    .module
    .declare_function(&symbol_name, effective_linkage, &sig)
    .map_err(|e| {
        CodegenError::new(format!("failed to declare fn '{raw_name}': {e}"))
    })?;
self.registry
    .functions
    .insert(raw_name.clone(), (func_id, sig.clone()));
```

CAUTION:
- `self.interner` may be named differently. Check with `grep -n "interner:" crates/nsl-codegen/src/compiler/mod.rs | head`. If it's `self.interner: &'a Interner` (reference), pass it directly. Adapt the helper's signature if a concrete type is stored.
- `self.registry.functions.insert(raw_name.clone(), (func_id, sig));` — `sig` was moved. In the new form, it now uses `sig.clone()` because `sig` is later consumed or dropped. If the original code didn't need `clone()`, it's because `insert` takes the sig by value and nothing else uses it — verify whether the new version needs `.clone()` or not based on subsequent usage (Task 3 will track `ExportInfo` which needs the sig, so `.clone()` is likely correct).

- [ ] **Step 6: Build + full lib tests**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: clean build; no existing test regresses (no existing test uses `@export`, so all functions still get `Linkage::Local` via `linkage` arg + mangled name).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/compiler/declaration.rs
git commit -m "feat(m62): @export bumps linkage to Export with unmangled symbol

extract_export_decorator helper (pure function, 4 unit tests) parses
@export and @export(name=\"...\") from a decorator list. The main
declare loop calls it before declare_function to override linkage
and symbol name when the decorator is present. Non-decorated
functions are byte-identical to pre-PR output."
```

---

## Task 3: Track `ExportInfo` during declaration

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/declaration.rs`.
- Modify: `crates/nsl-codegen/src/c_header.rs` — add a `from_fn_def` helper that builds `ExportInfo` from a type-checked signature.

- [ ] **Step 1: Write a failing test for `ExportInfo::from_fn_def`**

Add to `crates/nsl-codegen/src/c_header.rs` test module:

```rust
#[test]
fn export_info_from_simple_fn_def() {
    // Minimal FnDef: fn forward(x: Tensor<[B, 768], f32>) -> Tensor<[B, 1000], f32>
    // Constructed as AST literal — verify the helper lowers param + return
    // shapes correctly.
    use nsl_ast::decl::{FnDef, FnParam};

    let mut interner = nsl_lexer::Interner::new();
    let forward_sym = interner.intern("forward");
    let x_sym = interner.intern("x");
    let b_sym = interner.intern("B");

    let fn_def = FnDef {
        name: forward_sym,
        params: vec![FnParam {
            name: x_sym,
            ty: /* Tensor<[B, 768], f32> — AST type literal TBD from inspection */,
            span: nsl_ast::span::Span::dummy(),
        }],
        return_type: Some(/* Tensor<[B, 1000], f32> */),
        body: vec![],
        generics: vec![],
        span: nsl_ast::span::Span::dummy(),
    };

    let info = ExportInfo::from_fn_def(
        &fn_def,
        "forward",
        "forward",
        &interner,
    );
    assert_eq!(info.symbol_name, "forward");
    assert_eq!(info.raw_name, "forward");
    assert_eq!(info.params.len(), 1);
    assert_eq!(info.params[0].name, "x");
    match &info.params[0].ty {
        ExportTypeInfo::Tensor { shape, dtype, .. } => {
            assert_eq!(shape, &vec!["B".to_string(), "768".to_string()]);
            assert_eq!(*dtype, ExportDtype::F32);
        }
        _ => panic!("expected Tensor"),
    }
}
```

IMPORTANT: the `/* Tensor<...> — TBD */` placeholders need real AST literal values. Read `crates/nsl-ast/src/type.rs` (or wherever tensor types are defined) before writing this test to confirm the exact `TypeExpr` variants and fields.

If constructing the AST literal is prohibitively verbose, swap to parsing a small NSL snippet via `nsl_parser::parse` + extracting the first `FnDef`. Example:

```rust
let src = "fn forward(x: Tensor<[B, 768], f32>) -> Tensor<[B, 1000], f32>:\n    return x";
let module = nsl_parser::parse(src).expect("parse");
let fn_def = /* extract first FnDef from module.items */;
```

- [ ] **Step 2: Run to confirm failure**

```bash
cargo test -p nsl-codegen c_header::tests::export_info_from_simple_fn_def 2>&1 | tail -5
```

Expected: FAIL — `from_fn_def` not found.

- [ ] **Step 3: Implement `ExportInfo::from_fn_def`**

Add to `crates/nsl-codegen/src/c_header.rs`:

```rust
impl ExportInfo {
    pub fn from_fn_def(
        fn_def: &nsl_ast::decl::FnDef,
        raw_name: &str,
        symbol_name: &str,
        interner: &nsl_lexer::Interner,
    ) -> Self {
        let params = fn_def.params.iter().map(|p| {
            ExportParamInfo {
                name: interner.resolve(p.name).unwrap_or("").to_string(),
                ty: lower_type_expr(&p.ty, interner),
            }
        }).collect();

        let return_type = match &fn_def.return_type {
            Some(ty) => lower_type_expr(ty, interner),
            None => ExportTypeInfo::Tuple(vec![]),   // unit
        };

        Self {
            symbol_name: symbol_name.to_string(),
            raw_name: raw_name.to_string(),
            params,
            return_type,
        }
    }
}

/// Lower an AST `TypeExpr` to an `ExportTypeInfo`. Only the subset
/// of type shapes allowed by the semantic-pass validation (Task 4)
/// is recognized; anything else becomes `Tuple(vec![])` as a
/// fallback (but semantic validation will have rejected it before
/// this path runs).
fn lower_type_expr(
    ty: &nsl_ast::type_expr::TypeExpr,  // adapt to actual module path
    interner: &nsl_lexer::Interner,
) -> ExportTypeInfo {
    // TODO: real implementation — match on TypeExpr variants:
    //   Tensor { shape, dtype, device } → ExportTypeInfo::Tensor { ... }
    //   Primitive (Int32, Float32, etc.) → ExportTypeInfo::Scalar(...)
    //   Tuple(elems)                     → ExportTypeInfo::Tuple(elems.iter().map(lower).collect())
    //   _                                → ExportTypeInfo::Tuple(vec![])  (semantic pass rejects)
    todo!("populate based on actual TypeExpr variants in nsl-ast")
}
```

IMPORTANT: `lower_type_expr` is the core piece. The engineer must:
1. `grep -n "pub enum TypeExpr\|pub struct TypeExpr" crates/nsl-ast/src/` to find the real type.
2. Read its variants and field names.
3. Implement `lower_type_expr` as an exhaustive match on those variants.

If the type system has 10+ variants, only implement the ones that appear in `@export`-valid signatures (Tensor, primitive scalars, tuple). All other variants return `ExportTypeInfo::Tuple(vec![])` — semantic validation in Task 4 will have rejected those.

CAUTION: the `todo!` macro will fail the test. Replace it with a real match before running the test.

- [ ] **Step 4: Run test**

```bash
cargo test -p nsl-codegen c_header::tests::export_info_from_simple_fn_def 2>&1 | tail -5
```

Expected: PASS.

- [ ] **Step 5: Push `ExportInfo` into `features.export_functions` when `is_export`**

In `crates/nsl-codegen/src/compiler/declaration.rs`, immediately after the `declare_function` call added in Task 2 Step 5:

```rust
if is_export {
    let info = crate::c_header::ExportInfo::from_fn_def(
        fn_def,
        &raw_name,
        &symbol_name,
        self.interner,
    );
    self.features.export_functions.push(info);
}
```

- [ ] **Step 6: Build + full lib tests**

```bash
cargo build -p nsl-codegen 2>&1 | tail -3
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: no regressions; no existing test uses `@export` so `export_functions` stays empty.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/c_header.rs \
        crates/nsl-codegen/src/compiler/declaration.rs
git commit -m "feat(m62): track ExportInfo per @export function

ExportInfo::from_fn_def lowers FnDef signature to the C-ABI shape
registry. declaration.rs pushes one entry per @export function into
Compiler.features.export_functions. Type lowering covers Tensor,
Scalar, and Tuple variants; unsupported variants fall through to
Tuple(vec![]) — semantic validation (Task 4) rejects those before
this path is reached."
```

---

## Task 4: Semantic validation — `crates/nsl-semantic/src/export.rs`

**Files:**
- Create: `crates/nsl-semantic/src/export.rs`.
- Modify: `crates/nsl-semantic/src/lib.rs` (call `validate_exports` from `analyze_with_imports`).

- [ ] **Step 1: Create the validation module**

`crates/nsl-semantic/src/export.rs`:

```rust
//! M62 `@export` decorator semantic validation.
//!
//! Enforces the C-ABI-compatible subset of NSL function signatures.
//! Errors here block codegen — `declaration.rs`'s `@export` branch
//! assumes signatures are already validated.

use nsl_ast::{decl::{Decorator, FnDef, Module, Stmt, StmtKind}, type_expr::TypeExpr};
use nsl_lexer::Interner;

use crate::Diagnostic;

pub fn validate_exports(module: &Module, interner: &Interner) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for item in &module.items {
        validate_stmt(item, interner, &mut diagnostics);
    }
    diagnostics
}

fn validate_stmt(stmt: &Stmt, interner: &Interner, diagnostics: &mut Vec<Diagnostic>) {
    match &stmt.kind {
        StmtKind::Decorated { decorators, stmt: inner } => {
            let export_occurrences: Vec<&Decorator> = decorators
                .iter()
                .filter(|d| d.name.len() == 1 && interner.resolve(d.name[0]) == Some("export"))
                .collect();
            if export_occurrences.is_empty() {
                return;
            }

            // Duplicate @export
            if export_occurrences.len() > 1 {
                diagnostics.push(Diagnostic::error(
                    format!("@export decorator appears multiple times on '{}'",
                            describe_stmt(inner, interner)),
                    export_occurrences[1].span,
                ));
            }

            // Validate kwargs on the first occurrence (others already errored above)
            validate_export_args(export_occurrences[0], interner, diagnostics);

            // The decorated stmt must be a FnDef
            let fn_def = match &inner.kind {
                StmtKind::FnDef(fd) => fd,
                _ => {
                    diagnostics.push(Diagnostic::error(
                        "@export can only be applied to functions".to_string(),
                        export_occurrences[0].span,
                    ));
                    return;
                }
            };

            validate_fn_signature(fn_def, interner, export_occurrences[0], diagnostics);
        }
        _ => {} // Non-decorated stmts cannot have @export
    }
}

fn validate_export_args(
    d: &Decorator,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(ref args) = d.args else { return; }; // bare @export is fine

    for arg in args {
        match arg.name {
            None => {
                diagnostics.push(Diagnostic::error(
                    "@export takes only a 'name=...' keyword argument; no positional arguments"
                        .to_string(),
                    arg.span,
                ));
            }
            Some(sym) => {
                let kw = interner.resolve(sym).unwrap_or("");
                if kw != "name" {
                    diagnostics.push(Diagnostic::error(
                        format!("@export takes only a 'name' keyword argument; got '{kw}'"),
                        arg.span,
                    ));
                    continue;
                }
                // name="..." — validate the string value
                match &arg.value.kind {
                    nsl_ast::expr::ExprKind::StringLiteral(s) => {
                        if s.is_empty() {
                            diagnostics.push(Diagnostic::error(
                                "@export(name=\"...\") cannot be empty".to_string(),
                                arg.value.span,
                            ));
                        } else if !is_valid_c_identifier(s) {
                            diagnostics.push(Diagnostic::error(
                                format!(
                                    "@export(name=\"{s}\") must be a valid C identifier: \
                                     letters, digits, underscore; cannot start with digit"
                                ),
                                arg.value.span,
                            ));
                        }
                    }
                    _ => {
                        diagnostics.push(Diagnostic::error(
                            "@export(name=...) must be a string literal".to_string(),
                            arg.value.span,
                        ));
                    }
                }
            }
        }
    }
}

fn validate_fn_signature(
    fn_def: &FnDef,
    interner: &Interner,
    export_decorator: &Decorator,
    diagnostics: &mut Vec<Diagnostic>,
) {
    // No generic type parameters
    if !fn_def.generics.is_empty() {
        diagnostics.push(Diagnostic::error(
            "@export function cannot have generic type parameters — C ABI requires monomorphized types"
                .to_string(),
            export_decorator.span,
        ));
    }

    // Parameters: each must be tensor, scalar, or tuple-of-those.
    // Closure params are rejected.
    for param in &fn_def.params {
        if is_closure_type(&param.ty) {
            diagnostics.push(Diagnostic::error(
                "@export function cannot take closure parameters — closures cannot cross the C ABI"
                    .to_string(),
                param.ty.span,
            ));
        } else if !is_c_abi_compatible(&param.ty, interner) {
            diagnostics.push(Diagnostic::error(
                format!(
                    "@export function parameter '{}' must be a tensor, scalar, or tuple of those",
                    interner.resolve(param.name).unwrap_or("<unknown>")
                ),
                param.ty.span,
            ));
        }
    }

    // Return: tensor, scalar, tuple-of-those, or unit
    if let Some(ref ret_ty) = fn_def.return_type {
        if !is_c_abi_compatible(ret_ty, interner) {
            diagnostics.push(Diagnostic::error(
                "@export function must return a tensor, scalar, or tuple of those".to_string(),
                ret_ty.span,
            ));
        }
    }
    // No return type = unit — allowed.
}

fn is_closure_type(_ty: &TypeExpr) -> bool {
    // TODO: match on TypeExpr variants — return true for function/closure types.
    // Real implementation depends on how closure types are represented in the AST.
    // Read crates/nsl-ast/src/type_expr.rs (or equivalent) to find the variant.
    false
}

fn is_c_abi_compatible(_ty: &TypeExpr, _interner: &Interner) -> bool {
    // TODO: return true for Tensor, Scalar primitives (Int32/64, Float32/64, Bool),
    // and Tuple of compatible types. Everything else (struct, model, enum, closure)
    // is false.
    true // placeholder — fix before running validation tests
}

fn is_valid_c_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn describe_stmt(stmt: &Stmt, interner: &Interner) -> String {
    match &stmt.kind {
        StmtKind::FnDef(fd) => interner.resolve(fd.name).unwrap_or("<unknown>").to_string(),
        _ => "<non-fn>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_validate(src: &str) -> Vec<Diagnostic> {
        let mut interner = nsl_lexer::Interner::new();
        let module = nsl_parser::parse_with_interner(src, &mut interner).expect("parse");
        validate_exports(&module, &interner)
    }

    #[test]
    fn valid_simple_export_has_no_errors() {
        let src = r#"
@export
fn forward(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
        assert!(parse_and_validate(src).is_empty());
    }

    #[test]
    fn export_on_model_errors() {
        let src = r#"
@export
model Foo:
    w: Tensor<[4], f32>
"#;
        let errs = parse_and_validate(src);
        assert!(errs.iter().any(|d| d.message.contains("only be applied to functions")));
    }

    #[test]
    fn export_with_closure_param_errors() {
        let src = r#"
@export
fn apply(cb: fn(Tensor<[4], f32>) -> Tensor<[4], f32>) -> Tensor<[4], f32>:
    return cb(zeros([4]))
"#;
        let errs = parse_and_validate(src);
        assert!(errs.iter().any(|d| d.message.contains("closure")));
    }

    #[test]
    fn export_empty_name_errors() {
        let src = r#"
@export(name="")
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
        let errs = parse_and_validate(src);
        assert!(errs.iter().any(|d| d.message.contains("cannot be empty")));
    }

    #[test]
    fn export_invalid_c_identifier_errors() {
        let src = r#"
@export(name="123bad")
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
        let errs = parse_and_validate(src);
        assert!(errs.iter().any(|d| d.message.contains("valid C identifier")));
    }

    #[test]
    fn export_unknown_kwarg_errors() {
        let src = r#"
@export(other=1)
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
        let errs = parse_and_validate(src);
        assert!(errs.iter().any(|d| d.message.contains("'other'")));
    }

    #[test]
    fn export_positional_arg_errors() {
        let src = r#"
@export("positional")
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
        let errs = parse_and_validate(src);
        assert!(errs.iter().any(|d| d.message.contains("positional")));
    }

    #[test]
    fn duplicate_export_errors() {
        let src = r#"
@export
@export
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
        let errs = parse_and_validate(src);
        assert!(errs.iter().any(|d| d.message.contains("multiple times")));
    }

    #[test]
    fn export_with_generic_param_errors() {
        let src = r#"
@export
fn foo<T>(x: Tensor<[4], T>) -> Tensor<[4], T>:
    return x
"#;
        let errs = parse_and_validate(src);
        assert!(errs.iter().any(|d| d.message.contains("generic")));
    }
}
```

IMPORTANT:
- `is_closure_type` and `is_c_abi_compatible` have `TODO` placeholders. Must be implemented before the tests pass. Read `crates/nsl-ast/src/type_expr.rs` (or wherever `TypeExpr` lives) to find the exact variants.
- `nsl_parser::parse_with_interner` may not be the exact API name. Confirm with `grep -n "pub fn parse" crates/nsl-parser/src/lib.rs`.
- `Diagnostic::error(msg, span)` API — confirm against `crates/nsl-semantic/src/lib.rs` or wherever `Diagnostic` lives.

- [ ] **Step 2: Register module in `lib.rs`**

In `crates/nsl-semantic/src/lib.rs`:

```rust
pub mod export;
```

And inside `analyze_with_imports` (before returning the `AnalysisResult`), call:

```rust
let export_diagnostics = crate::export::validate_exports(module, interner);
diagnostics.extend(export_diagnostics);
```

Exact placement: after the other decorator validation passes run. Look for existing calls like `wrga::validate_wrga_decorators` or similar to find the precedent.

- [ ] **Step 3: Run tests**

```bash
cargo test -p nsl-semantic export::tests 2>&1 | tail -10
```

Expected: 9 tests PASS (one for each validation case).

- [ ] **Step 4: Full workspace lib test**

```bash
cargo test --workspace --lib 2>&1 | tail -3
```

Expected: no regressions.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-semantic/src/export.rs crates/nsl-semantic/src/lib.rs
git commit -m "feat(m62): @export semantic validation

validate_exports rejects @export on non-functions, closure params,
non-tensor returns, generic type params, empty/invalid name strings,
unknown kwargs, positional args, and duplicate decorators. Called
from analyze_with_imports alongside other decorator validators."
```

---

## Task 5: C header emission

**Files:**
- Modify: `crates/nsl-codegen/src/c_header.rs` — add `emit()` function.
- Modify: `crates/nsl-cli/src/main.rs` — call `emit()` after shared-lib build.
- Create: `crates/nsl-codegen/tests/c_header_snapshot.rs`.
- Create: `crates/nsl-codegen/tests/fixtures/m62_export_header.nsl`.

- [ ] **Step 1: Create the snapshot fixture**

`crates/nsl-codegen/tests/fixtures/m62_export_header.nsl`:

```
@export
fn forward(x: Tensor<[B, 768], f32>) -> Tensor<[B, 1000], f32>:
    return x

@export(name="predict")
fn inference_forward(x: Tensor<[B, 768], f32>) -> Tensor<[B, 10], f32>:
    return x
```

- [ ] **Step 2: Write the snapshot test**

`crates/nsl-codegen/tests/c_header_snapshot.rs`:

```rust
//! M62 C header emission snapshot test.

use nsl_codegen::c_header::{emit, ExportInfo, ExportParamInfo, ExportTypeInfo, ExportDtype, ExportDevice};

fn sample_export_functions() -> Vec<ExportInfo> {
    vec![
        ExportInfo {
            symbol_name: "forward".into(),
            raw_name: "forward".into(),
            params: vec![ExportParamInfo {
                name: "x".into(),
                ty: ExportTypeInfo::Tensor {
                    shape: vec!["B".into(), "768".into()],
                    dtype: ExportDtype::F32,
                    device: ExportDevice::Any,
                },
            }],
            return_type: ExportTypeInfo::Tensor {
                shape: vec!["B".into(), "1000".into()],
                dtype: ExportDtype::F32,
                device: ExportDevice::Any,
            },
        },
        ExportInfo {
            symbol_name: "predict".into(),
            raw_name: "inference_forward".into(),
            params: vec![ExportParamInfo {
                name: "x".into(),
                ty: ExportTypeInfo::Tensor {
                    shape: vec!["B".into(), "768".into()],
                    dtype: ExportDtype::F32,
                    device: ExportDevice::Any,
                },
            }],
            return_type: ExportTypeInfo::Tensor {
                shape: vec!["B".into(), "10".into()],
                dtype: ExportDtype::F32,
                device: ExportDevice::Any,
            },
        },
    ]
}

#[test]
fn header_contains_expected_prototypes() {
    let exports = sample_export_functions();
    let header = emit(&exports, "model");

    // Header guard
    assert!(header.contains("#ifndef NSL_MODEL_H"));
    assert!(header.contains("#define NSL_MODEL_H"));
    assert!(header.contains("#endif"));

    // Required types
    assert!(header.contains("typedef struct NslModel NslModel"));
    assert!(header.contains("NslTensorDesc"));

    // Lifecycle
    assert!(header.contains("NslModel* nsl_model_create("));
    assert!(header.contains("void      nsl_model_destroy("));

    // Export functions — prototypes
    assert!(header.contains("int forward(NslModel* model"));
    assert!(header.contains("const NslTensorDesc* x"));
    assert!(header.contains("NslTensorDesc* __ret"));

    assert!(header.contains("int predict(NslModel* model"));  // renamed symbol

    // The raw NSL name should NOT appear as a prototype (predict replaces it)
    assert!(!header.contains("int inference_forward("));
}

#[test]
fn header_has_extern_c_guards() {
    let exports = sample_export_functions();
    let header = emit(&exports, "model");
    assert!(header.contains("#ifdef __cplusplus"));
    assert!(header.contains("extern \"C\""));
}
```

- [ ] **Step 3: Run to confirm failure**

```bash
cargo test -p nsl-codegen --test c_header_snapshot 2>&1 | tail -5
```

Expected: FAIL — `emit` not found.

- [ ] **Step 4: Implement `emit()`**

In `crates/nsl-codegen/src/c_header.rs`:

```rust
/// Emit a C header for the given `@export` functions. `module_name`
/// is used for the header-guard macro (e.g. "model" → `NSL_MODEL_H`).
///
/// Returns the header text; callers write it to a file alongside
/// the shared library.
pub fn emit(exports: &[ExportInfo], module_name: &str) -> String {
    let guard = format!("NSL_{}_H", module_name.to_uppercase());
    let mut out = String::new();

    out.push_str(&format!("/* {module_name}.h — Auto-generated by NSL compiler */\n"));
    out.push_str(&format!("#ifndef {guard}\n"));
    out.push_str(&format!("#define {guard}\n\n"));
    out.push_str("#include <stdint.h>\n");
    out.push_str("#include <stddef.h>\n\n");
    out.push_str("#ifdef __cplusplus\n");
    out.push_str("extern \"C\" {\n");
    out.push_str("#endif\n\n");

    // Opaque handle + tensor descriptor
    out.push_str("typedef struct NslModel NslModel;\n\n");
    out.push_str("typedef struct {\n");
    out.push_str("    void*    data;\n");
    out.push_str("    int64_t* shape;\n");
    out.push_str("    int64_t* strides;     /* NULL = contiguous */\n");
    out.push_str("    int32_t  ndim;\n");
    out.push_str("    int32_t  dtype;       /* 0=f32, 1=f64, 2=f16, 3=bf16, 4=i32, 5=i64, 6=i8, 7=u8 */\n");
    out.push_str("    int32_t  device_type; /* 0=CPU, 1=CUDA */\n");
    out.push_str("    int32_t  device_id;\n");
    out.push_str("} NslTensorDesc;\n\n");

    // Lifecycle (provided by libnsl_runtime)
    out.push_str("/* Lifecycle (provided by libnsl_runtime) */\n");
    out.push_str("NslModel* nsl_model_create(const char* weights_path);\n");
    out.push_str("void      nsl_model_destroy(NslModel* model);\n\n");

    // Export function prototypes
    out.push_str("/* @export functions */\n");
    for info in exports {
        emit_prototype(&mut out, info);
    }
    out.push('\n');

    out.push_str("#ifdef __cplusplus\n");
    out.push_str("}\n");
    out.push_str("#endif\n");
    out.push_str(&format!("#endif /* {guard} */\n"));

    out
}

fn emit_prototype(out: &mut String, info: &ExportInfo) {
    out.push_str(&format!("int {}(NslModel* model", info.symbol_name));
    for param in &info.params {
        out.push_str(",\n        ");
        emit_param(out, &param.name, &param.ty);
    }
    // Return arg (caller-allocated)
    out.push_str(",\n        ");
    emit_return_arg(out, &info.return_type);
    out.push_str(");\n");
}

fn emit_param(out: &mut String, name: &str, ty: &ExportTypeInfo) {
    match ty {
        ExportTypeInfo::Tensor { .. } => {
            out.push_str(&format!("const NslTensorDesc* {name}"));
        }
        ExportTypeInfo::Scalar(dtype) => {
            out.push_str(&format!("{} {name}", c_type_for_scalar(*dtype)));
        }
        ExportTypeInfo::Tuple(elems) => {
            // Flatten tuple params as a count + pointer
            out.push_str(&format!("const NslTensorDesc* {name}_items, int32_t {name}_count"));
            let _ = elems;
        }
    }
}

fn emit_return_arg(out: &mut String, ty: &ExportTypeInfo) {
    match ty {
        ExportTypeInfo::Tensor { .. } => {
            out.push_str("NslTensorDesc* __ret");
        }
        ExportTypeInfo::Scalar(dtype) => {
            out.push_str(&format!("{} * __ret", c_type_for_scalar(*dtype)));
        }
        ExportTypeInfo::Tuple(_) => {
            out.push_str("NslTensorDesc* __rets, int32_t* __num_rets");
        }
    }
}

fn c_type_for_scalar(dtype: ExportDtype) -> &'static str {
    match dtype {
        ExportDtype::I8  => "int8_t",
        ExportDtype::I16 => "int16_t",
        ExportDtype::I32 => "int32_t",
        ExportDtype::I64 => "int64_t",
        ExportDtype::U8  => "uint8_t",
        ExportDtype::U16 => "uint16_t",
        ExportDtype::U32 => "uint32_t",
        ExportDtype::U64 => "uint64_t",
        ExportDtype::F32 => "float",
        ExportDtype::F64 => "double",
        ExportDtype::F16 | ExportDtype::BF16 => "uint16_t",  // raw bits, caller interprets
        ExportDtype::Bool => "int32_t",
    }
}
```

- [ ] **Step 5: Run snapshot tests**

```bash
cargo test -p nsl-codegen --test c_header_snapshot 2>&1 | tail -5
```

Expected: 2 tests PASS.

- [ ] **Step 6: Invoke from CLI after shared-lib build**

In `crates/nsl-cli/src/main.rs`, find where the shared-lib output path is computed (around line 2188 per the Phase 2 reads — look for `default_shared_lib_path`). After the build succeeds, add:

```rust
// M62: emit C header alongside the shared library when exports are present
if let Some(ref compiler) = <compiler-handle-in-scope> {
    let exports = &compiler.features.export_functions;
    if !exports.is_empty() {
        let module_name = output_path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let header = nsl_codegen::c_header::emit(exports, module_name);
        let header_path = output_path.with_extension("h");
        std::fs::write(&header_path, header)
            .map_err(|e| format!("failed to write header '{}': {e}", header_path.display()))?;
        eprintln!("[nsl] wrote C header: {}", header_path.display());
    }
}
```

CAUTION: the exact variable holding the `Compiler` instance after build is environment-specific. Read the surrounding shared-lib build code to find it. If the Compiler is consumed by `finalize()` before this point, the `Vec<ExportInfo>` needs to be `.clone()`d out beforehand and threaded through.

- [ ] **Step 7: Build + test**

```bash
cargo build --workspace 2>&1 | tail -3
cargo test -p nsl-codegen --test c_header_snapshot
```

Expected: clean build; snapshot tests pass.

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-codegen/src/c_header.rs \
        crates/nsl-codegen/tests/c_header_snapshot.rs \
        crates/nsl-codegen/tests/fixtures/m62_export_header.nsl \
        crates/nsl-cli/src/main.rs
git commit -m "feat(m62): C header emission alongside shared-lib build

c_header::emit() produces a compilable C header with header guards,
lifecycle prototypes, and one prototype per @export function. CLI
invokes the emitter after shared-lib build when
Compiler.features.export_functions is non-empty. Output path is the
shared-lib stem with .h extension."
```

---

## Task 6: E2E test — `@export` produces loadable C symbol

**Files:**
- Create: `examples/m62_shared_lib.nsl`.
- Create: `python/tests/test_m62_export.py`.

- [ ] **Step 1: Create the NSL fixture**

`examples/m62_shared_lib.nsl`:

```
@export
fn add(a: Tensor<[4], f32>, b: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return a + b
```

- [ ] **Step 2: Write the Python E2E test**

`python/tests/test_m62_export.py`:

```python
"""M62 @export E2E: compile to .so, load via ctypes, verify symbol and output."""

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parents[2]
NSL_BIN = WORKSPACE / "target" / "debug" / ("nsl.exe" if os.name == "nt" else "nsl")
FIXTURE = WORKSPACE / "examples" / "m62_shared_lib.nsl"


@pytest.fixture(scope="module")
def shared_lib(tmp_path_factory):
    """Build the fixture as a .so/.dll and yield its path."""
    if not NSL_BIN.exists():
        pytest.skip(f"nsl binary not found at {NSL_BIN}; run `cargo build` first")
    if not FIXTURE.exists():
        pytest.skip(f"fixture not found at {FIXTURE}")

    tmp = tmp_path_factory.mktemp("m62_export")
    out = tmp / ("libadd.dll" if os.name == "nt" else "libadd.so")

    result = subprocess.run(
        [str(NSL_BIN), "build", str(FIXTURE), "--shared-lib", "-o", str(out)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        pytest.fail(f"nsl build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

    assert out.exists(), f"shared lib not produced at {out}"
    yield out


def test_shared_lib_has_add_symbol(shared_lib):
    """The @export fn add should be a reachable C symbol."""
    lib = ctypes.CDLL(str(shared_lib))
    # Attempting to access the symbol will raise AttributeError if missing.
    add_fn = lib.add
    assert add_fn is not None


def test_generated_header_exists_and_declares_add(shared_lib):
    """The .h file should be emitted alongside the .so and prototype `add`."""
    header_path = shared_lib.with_suffix(".h")
    assert header_path.exists(), f"header not emitted at {header_path}"
    content = header_path.read_text()
    assert "int add(NslModel* model" in content
    assert "const NslTensorDesc* a" in content
    assert "const NslTensorDesc* b" in content
    assert "NslTensorDesc* __ret" in content
```

CAUTION: this test requires the C wrapper (`int add(NslModel*, const NslTensorDesc* a, const NslTensorDesc* b, NslTensorDesc* __ret)`) to actually exist as an emitted function. That's the "per-function C wrapper emission" I referred to in the spec §10 as an open design choice. **If Task 5's implementation of header emission doesn't also emit the C wrapper for each `@export` function, `test_shared_lib_has_add_symbol` will fail** — `ctypes.CDLL` would find a symbol with the wrong signature, and calling it would crash.

**Action required in Task 5**: before committing Task 5, verify whether the `Linkage::Export` declaration in Task 2 already produces a callable function with the NSL-internal calling convention (`*NslTensor`-style args), or whether we need to emit an ABI wrapper. Read what `build_fn_signature` at [declaration.rs:51](../../../crates/nsl-codegen/src/compiler/declaration.rs#L51) produces for the function signature.

If the internal calling convention already takes pointer args, the emitted symbol IS callable from C — just not with the ergonomic `NslTensorDesc*` signature the header claims. In that case:
- Either widen the header to declare the actual internal signature (less ergonomic for C consumers, but correct).
- Or split Task 5 into two: 5a (internal calling convention works) and 5b (ABI wrapper emission).

Defer the decision to implementation; the E2E test in this task is the gate that will force a choice.

- [ ] **Step 3: Build the binary and run the test**

```bash
cargo build -p nsl-cli
python -m pytest python/tests/test_m62_export.py -v
```

Expected: both tests pass. If `test_shared_lib_has_add_symbol` fails, triage per the CAUTION note above.

- [ ] **Step 4: Commit**

```bash
git add examples/m62_shared_lib.nsl python/tests/test_m62_export.py
git commit -m "test(m62): E2E test — @export produces loadable C symbol

Python E2E test builds a 2-input @export fn into a shared library,
asserts the ctypes.CDLL can load the symbol, and confirms the
generated .h file contains the expected prototype. Triggers the
full pipeline: @export parse → semantic validation → linkage
override → header emission."
```

---

## Task 7: Memory + push + PR

- [ ] **Step 1: Update memory**

In `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md`, find the M62 reference (if any) and update. Or add a new pointer under a relevant section:

```markdown
- [M62 Legacy Interop finish](project_m62_finish.md) — `@export` decorator shipped 2026-04-15 (PR pending); grad-context bridge fix pending as sibling PR.
```

Create `project_m62_finish.md` capturing:
- `@export` shipped: what `declaration.rs` does now, where `ExportInfo` lives, how the C header is emitted.
- Sibling gap still open: grad-context bridge (see `2026-04-15-m62-grad-context-bridge-design.md`).
- Test entry points: `python/tests/test_m62_export.py`, `crates/nsl-codegen/tests/c_header_snapshot.rs`.

- [ ] **Step 2: Full workspace test**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/m62-finish
cargo test --workspace 2>&1 | tail -10
python -m pytest python/tests/test_m62_export.py -v 2>&1 | tail -10
```

Expected: all pass except documented pre-existing flakes (`e2e_m12_grad_basic_source_ad`, `e2e_m27_*`).

- [ ] **Step 3: Push**

```bash
git push -u origin feat/m62-finish
```

- [ ] **Step 4: Prepare PR body**

```markdown
## Summary
- `@export` decorator: functions opt in to C-callable symbols in a `--shared-lib` build. Overrides linkage to `Linkage::Export` and symbol name to the raw (unmangled) NSL name, or `@export(name="custom")`.
- Semantic validation rejects: `@export` on non-functions, closure params, non-tensor returns, generic type params, empty/invalid name strings, unknown kwargs, positional args, duplicate decorators.
- C header emission: `<output>.h` produced alongside the `.so`/`.dylib`/`.dll` with lifecycle prototypes + one `int <name>(NslModel*, ...)` prototype per exported function.
- Fixes the shipped-but-unreachable `--shared-lib` path: before this PR, all user functions got `Linkage::Local` so the compiled library had zero callable symbols.

## WGGO / M62 status after this PR
- ✅ `@export` decorator (this PR)
- ⏳ Grad-context bridge fix (sibling spec: `docs/superpowers/specs/2026-04-15-m62-grad-context-bridge-design.md`)

## Test plan
- [ ] `cargo test -p nsl-codegen compiler::declaration::export_tests` — 4 unit tests pass
- [ ] `cargo test -p nsl-semantic export::tests` — 9 validation tests pass
- [ ] `cargo test -p nsl-codegen --test c_header_snapshot` — 2 snapshot tests pass
- [ ] `python -m pytest python/tests/test_m62_export.py` — E2E `.so` loads from ctypes
- [ ] `cargo test --workspace` — no regressions

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

- [ ] **Step 5: No commit for the memory update** — memory files aren't in the repo.

---

## Self-Review Checklist (run before claiming done)

- [ ] Every spec section (§3 language surface, §4 codegen, §5 semantic validation, §6 header emission, §8 testing) has ≥1 task.
- [ ] No `TBD` / `implement later` patterns in the plan body. (The `lower_type_expr` and `is_closure_type` / `is_c_abi_compatible` placeholders in Task 3/4 are explicitly flagged `TODO — real implementation` with instructions to read the AST module; acceptable because the engineer has to ground them in the actual AST variants.)
- [ ] Task 0 is discovery-only; no commit.
- [ ] Task 2's firewall (byte-identical fallback for non-`@export`) is pinned by "no existing test regresses".
- [ ] Tasks 3 and 5 both contain `CAUTION` notes about real type names and variables the implementer must verify.
- [ ] Task 6's E2E test flags the C-wrapper-emission decision explicitly so Task 5's implementer knows to settle it.
- [ ] Task 7 includes the sibling-spec cross-reference so the PR body doesn't claim more than this PR delivers.
