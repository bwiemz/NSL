<!-- owner: @bwiemz -->

# Adding a Language Feature

End-to-end walkthrough of shipping a new language construct. Uses the real `@export` decorator from M62 (PR #45, merged 2026-04-15) as the worked example — scoped enough to fit in one sitting, broad enough to touch every compiler stage that matters.

## What `@export` does

`@export` opts a function into C-ABI linkage so host languages (Python, C++) can call into a compiled NSL shared library. Without it, every function compiles to `Linkage::Local` and is invisible outside the binary.

```python
@export
fn add(a: Tensor<[4], f32>, b: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return a + b

@export(name="predict")
fn inference_forward(x: Tensor<[B, D], f32>) -> Tensor<[B, C], f32>:
    return model(x)
```

The first form exports under the function's own name; the second overrides the C symbol name. Both forms produce an entry in the generated `.h` header alongside the compiled `.so`/`.dylib`/`.dll`.

## The walkthrough

Each stop shows the actual diff from PR #45 (merge commit `112a7f28`). Commits are presented in implementation order; each was individually green.

### 1. Spec delta

File: [`spec/11-interoperability.nsl.md`](../../spec/11-interoperability.nsl.md)

PR #45 did not modify the main spec file directly. The feature decision lives in a design doc created in the first commit of the branch:

File: [`docs/superpowers/specs/2026-04-15-m62-export-decorator-design.md`](../../docs/superpowers/specs/2026-04-15-m62-export-decorator-design.md)

```
## 1. Goal
Make `nsl build --shared-lib model.nsl` produce a `.so`/`.dylib`/`.dll`
with C-callable symbols. Today, all user functions get `Linkage::Local`
unconditionally at declaration.rs:29 regardless of the `--shared-lib`
flag, so the compiled library has zero reachable entry points.

Policy: strict opt-in — only `@export` functions become C-callable.
Everything else stays `Linkage::Local`.
```

**Why first:** the spec is the contract. It identified the root problem (silent broken behavior — `--shared-lib` compiled fine but exported nothing), established the strict opt-in policy, and declared that no parser or AST changes were needed. Everything downstream followed from those three decisions.

### 2. Lexer

No lexer changes. `@` followed by an identifier was already legal decorator syntax. `@export` is lexically indistinguishable from `@no_grad` or any other decorator — the lexer emits `At` + `Ident("export")` and stops there. No new token type was needed.

### 3. AST

No AST changes. The generic `Decorator { name: Vec<Symbol>, args: Option<Vec<Arg>>, span }` node produced by the existing parser already represents `@export` and `@export(name="...")` correctly. The design doc explicitly called this out:

```
No parser or AST changes. The generic decorator infrastructure at
parser/stmt.rs:428 already produces Decorator { name, args, span }
for @export and @export(name="...").
```

Lesson: before adding a new AST variant, check whether the existing catch-all infrastructure already represents the construct. Here, it did.

### 4. Parser

No parser changes for the same reason as AST. The existing decorator-parse path (at `crates/nsl-parser/src/stmt.rs`) already handles the `@ident` and `@ident(kwarg=value)` syntax and attaches the decorators to the following `fn` statement. The `@export`-specific meaning is entirely resolved downstream, in semantic and codegen.

### 5. Semantic

File: [`crates/nsl-semantic/src/export.rs`](../../crates/nsl-semantic/src/export.rs) (new file)
File: [`crates/nsl-semantic/src/lib.rs`](../../crates/nsl-semantic/src/lib.rs) (wiring)

Commit `ea737fb0` introduced `validate_exports`, a pure-additive pass that appends diagnostics without modifying other analysis state.

**New file `export.rs` — entry point:**

```rust
/// Run `@export` validation over the top-level statements of a module.
///
/// Returns diagnostics that should be appended to the rest of the
/// analysis diagnostics.  Pure-additive: does not read or modify
/// other analysis state.
pub fn validate_exports(module: &Module, interner: &Interner) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for stmt in &module.stmts {
        validate_stmt(stmt, interner, &mut diagnostics);
    }
    diagnostics
}

fn validate_stmt(stmt: &Stmt, interner: &Interner, diagnostics: &mut Vec<Diagnostic>) {
    let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind else {
        return;
    };

    let export_occurrences: Vec<&Decorator> = decorators
        .iter()
        .filter(|d| {
            d.name.len() == 1
                && interner.resolve(d.name[0].0).unwrap_or("") == "export"
        })
        .collect();

    if export_occurrences.is_empty() { return; }

    // Duplicate @export
    if export_occurrences.len() > 1 {
        diagnostics.push(
            Diagnostic::error(format!(
                "@export decorator appears multiple times on '{}'",
                describe_stmt(inner, interner)
            ))
            .with_label(export_occurrences[1].span, "duplicate @export"),
        );
    }
    // ...continues: validates kwargs, checks inner is a FnDef,
    //   checks arg/return types are C-ABI compatible, etc.
```

**Wiring in `lib.rs`:**

```rust
+    // M62: Run `@export` decorator validation.  Pure-additive — appends
+    // diagnostics without touching other analysis state.
+    diagnostics.extend(crate::export::validate_exports(module, interner));
```

Semantic validation enforces nine rules: the decorator must be on a `fn` (not a `model`, closure, or anything else); no positional args; only `name` kwarg; `name` must be a non-empty string literal; no duplicate `@export` on the same function; all param types must be C-ABI-compatible tensors or scalars; return type must not use generic type params. Errors here block codegen — the codegen branch assumes all `@export` signatures are already validated.

### 6. Codegen — linkage override

File: [`crates/nsl-codegen/src/compiler/declaration.rs`](../../crates/nsl-codegen/src/compiler/declaration.rs)

Commit `5e15c9b6` — the linkage override. The change is surgical: a pure helper `extract_export_decorator` parses the decorator list, then the main declare loop checks its result before calling `declare_function`.

```rust
+            // M62: @export override — if decorated, use Export linkage and an
+            // unmangled (or user-named) symbol so C consumers see a clean ABI.
+            let (is_export, override_name) = match decorators {
+                Some(decos) => extract_export_decorator(decos, self.interner),
+                None => (false, None),
+            };
+            let effective_linkage = if is_export {
+                Linkage::Export
+            } else {
+                linkage
+            };
+            let symbol_name = if is_export {
+                override_name.unwrap_or_else(|| raw_name.clone())
+            } else {
+                cranelift_name.clone()
+            };
+
             let func_id = self
                 .module
-                .declare_function(&cranelift_name, linkage, &sig)
+                .declare_function(&symbol_name, effective_linkage, &sig)
```

The `extract_export_decorator` helper is a pure function (no `&mut self`) so it can be unit-tested independently of a full `Compiler`:

```rust
fn extract_export_decorator(
    decorators: &[nsl_ast::decl::Decorator],
    interner: &nsl_lexer::Interner,
) -> (bool, Option<String>) {
    for d in decorators {
        if d.name.len() != 1 { continue; }
        let dname = interner.resolve(d.name[0].0).unwrap_or("");
        if dname != "export" { continue; }
        // found — extract optional name="..." kwarg
        if let Some(ref args) = d.args {
            for arg in args {
                if let Some(name_sym) = arg.name {
                    let arg_name = interner.resolve(name_sym.0).unwrap_or("");
                    if arg_name == "name" {
                        if let ExprKind::StringLiteral(s) = &arg.value.kind {
                            return (true, Some(s.clone()));
                        }
                    }
                }
            }
        }
        return (true, None);
    }
    (false, None)
}
```

Non-decorated functions are byte-identical to pre-PR output — the two branches collapse.

### 7. Codegen — type model and header emission

Files:
- [`crates/nsl-codegen/src/c_header.rs`](../../crates/nsl-codegen/src/c_header.rs) (new file, commits `c3060875` + `0dfaa941` + `46239372`)
- [`crates/nsl-codegen/src/compiler/mod.rs`](../../crates/nsl-codegen/src/compiler/mod.rs) (new field)
- [`crates/nsl-codegen/src/lib.rs`](../../crates/nsl-codegen/src/lib.rs) (re-export)

**Type model** (`c3060875`): `ExportInfo` and `ExportTypeInfo` capture each `@export` function's C-ABI-compatible shape. `FeatureConfigs` in `compiler/mod.rs` gets a new `export_functions: Vec<ExportInfo>` field that declaration.rs populates and that header emission reads.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportInfo {
    pub symbol_name: String,   // appears in .so export table
    pub raw_name:    String,   // NSL-side name (for diagnostics)
    pub params:      Vec<ExportParamInfo>,
    pub return_type: ExportTypeInfo,
}

pub enum ExportTypeInfo {
    Tensor { shape: Vec<String>, dtype: ExportDtype, device: ExportDevice },
    Scalar(ExportDtype),
    Tuple(Vec<ExportTypeInfo>),
}
```

**Header emission** (`46239372`): `c_header::emit()` turns the `Vec<ExportInfo>` into a compilable `.h` file with header guards, `NslTensorDesc` typedef, lifecycle prototypes, and one prototype per exported function:

```rust
pub fn emit(exports: &[ExportInfo], module_name: &str) -> String {
    let guard = format!("NSL_{}_H", sanitize_header_guard(module_name));
    let mut out = String::new();
    out.push_str(&format!("#ifndef {guard}\n#define {guard}\n\n"));
    out.push_str("#include <stdint.h>\n#include <stddef.h>\n\n");
    // ... NslTensorDesc typedef, lifecycle prototypes ...
    out.push_str("/* @export functions */\n");
    for info in exports { emit_prototype(&mut out, info); }
    out.push_str(&format!("#endif /* {guard} */\n"));
    out
}
```

The CLI (`crates/nsl-cli/src/main.rs`) invokes `emit()` after the shared-lib build when `export_functions` is non-empty, writing the header to the same directory with the `.h` extension.

### 8. Linker

File: [`crates/nsl-codegen/src/linker.rs`](../../crates/nsl-codegen/src/linker.rs)

Commit `233f61b0` found a platform quirk: on MSVC, Cranelift's `Linkage::Export` alone is not sufficient — `link.exe /DLL` still needs an explicit `/EXPORT:<sym>` flag for the symbol to appear in the DLL export table (verified via `dumpbin`). On Unix, `Linkage::Export` is sufficient.

```rust
+pub fn link_shared_with_exports(
+    obj_paths: &[PathBuf],
+    output_path: &Path,
+    extra_exports: &[&str],
+) -> Result<(), CodegenError> {
     if cfg!(target_os = "windows") {
-        link_shared_msvc(obj_paths, output_path, &runtime_lib)
+        link_shared_msvc(obj_paths, output_path, &runtime_lib, extra_exports)
             .or_else(|_| link_shared_gcc(obj_paths, output_path, &runtime_lib))
     } else {
         link_shared_gcc(obj_paths, output_path, &runtime_lib)
```

A second fix: `/EXPORT:main` was previously always emitted but `pure-@export` sources have no `main` symbol, causing a linker error. The fix probes the object file bytes to confirm `main` exists before adding the flag.

### 9. Runtime

No runtime changes. The `nsl_model_create` / `nsl_model_destroy` lifecycle functions already shipped as part of `crates/nsl-runtime/src/c_api.rs`. Symbol resolution happens at link time via the existing `libnsl_runtime` that the shared-lib build always links in.

### 10. Tests

Files:
- Snapshot: [`crates/nsl-codegen/tests/c_header_snapshot.rs`](../../crates/nsl-codegen/tests/c_header_snapshot.rs)
- Fixture: [`crates/nsl-codegen/tests/fixtures/m62_export_header.nsl`](../../crates/nsl-codegen/tests/fixtures/m62_export_header.nsl)
- E2E example: [`examples/m62_shared_lib.nsl`](../../examples/m62_shared_lib.nsl)
- Python E2E: [`python/tests/test_m62_export.py`](../../python/tests/test_m62_export.py)

**Example NSL source** (the simplest possible `@export` program):

```python
@export
fn add(a: Tensor<[4], f32>, b: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return a + b
```

**Snapshot test** — pins the exact C header output for two `@export` functions:

```rust
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
            return_type: ExportTypeInfo::Tensor { /* ... */ },
        },
        // second function with symbol_name="predict", raw_name="inference_forward"
    ]
}
// insta::assert_snapshot!(emit(&sample_export_functions(), "model"));
```

**Python E2E test** — verifies the `.so` symbol is loadable via `ctypes`:

```python
def test_shared_lib_has_add_symbol(shared_lib):
    """The @export fn add should be a reachable C symbol."""
    lib = ctypes.CDLL(str(shared_lib))
    sym = getattr(lib, "add", None)
    assert sym is not None, f"'add' symbol not found in {shared_lib}"
```

## PR structure

PR #45 shipped as 9 commits in this order:

1. `ced1d69b` — design doc
2. `0ccc1448` — implementation plan
3. `6c6bd0b2` — grad-context bridge design (separate fix, same branch)
4. `c3060875` — `ExportInfo` + `ExportTypeInfo` type model (no behavior yet)
5. `5e15c9b6` — linkage override in `declaration.rs` + 4 unit tests
6. `0dfaa941` — `ExportInfo::from_fn_def` type lowering + tracking
7. `ea737fb0` — semantic validation (9 rules)
8. `46239372` — C header emission + snapshot test
9. `233f61b0` — linker fixes + Python E2E test

Commit 4 (`ExportInfo` type model) is a useful pattern: define the data shape first, with the consuming field initialized to empty, so downstream commits have something concrete to fill in and read from. It also gives you a pinnable unit test for the struct shape before any behavior exists.

## Recurring compile-error traps

Adding a field to `Compiler`, `FeatureConfigs`, or similar central structs triggers these for every initializer site across the codebase:

- **E0063** — "missing fields `<name>` in initializer of `compiler::Compiler<'_>`". Every `Compiler::new` call needs the new field. Run `cargo check 2>&1 | grep "E0063"` to find all sites.
- **E0599** — "no method named `<x>` found for `&mut compiler::Compiler<'_>`". Usually a method you referenced but forgot to add; check the `impl Compiler` block.
- **E0308** — mismatched types. Usually at FFI boundaries where `&str` vs `String` or `Vec<T>` vs `&[T]` diverge.

Expect 10–100+ of these when extending the parser or codegen. Follow the compiler's suggestions — they're almost always right. The pattern from PR #45 (adding `export_functions: Vec<ExportInfo>` to `FeatureConfigs`) triggered exactly this: every `FeatureConfigs { .. }` initializer needed the new field or a `..Default::default()` tail.

**Linker surprises:** If your new feature produces a symbol that must be visible from a DLL on Windows, `Linkage::Export` in Cranelift is necessary but not sufficient — you also need an explicit `/EXPORT:<sym>` linker flag. See `link_shared_with_exports()` in `linker.rs` for the pattern.

See [Testing-Strategy](Testing-Strategy.md) for the test types referenced above.

---

*Last structurally verified against commit `112a7f28` on 2026-04-21 (walkthrough anchored to PR #45, merge commit `112a7f28`).*
