# M13: Import System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Complete the file-based module system so NSL code can import from a standard library and other files, unblocking M14's train block + optimizers written in NSL.

**Architecture:** Multi-file compilation pipeline with strict pass separation (parse all → semantic all → codegen all). Modules resolved via `NSL_STDLIB_PATH` env var or `<exe_dir>/stdlib/` fallback. Name mangling (`nsl_math__clamp`) prevents Cranelift symbol collisions. The `import X as Y` alias form enables `math.clamp(...)` qualified access.

**Tech Stack:** Rust (nsl-cli, nsl-codegen, nsl-parser, nsl-semantic, nsl-ast crates), Cranelift 0.116

---

## Current State

**What already exists (extensive — M13 is mostly gap-filling):**

- **Parser:** `parse_import_stmt()` and `parse_from_import_stmt()` in `crates/nsl-parser/src/decl.rs:379-484` — handles `from X import a, b`, `import X.Y`, `import X.Y.*`, `import X.Y.{a, b}`, aliases with `as`
- **AST:** `ImportStmt`, `FromImportStmt`, `ImportItems` (Module/Named/Glob), `ImportItem` with alias in `crates/nsl-ast/src/decl.rs:102-128`
- **Resolver:** `resolve_import()` in `crates/nsl-cli/src/resolver.rs` — resolves import paths relative to importing file's directory
- **Loader:** `load_all_modules()` in `crates/nsl-cli/src/loader.rs:127-244` — full 3-phase pipeline: parse worklist → topo sort with cycle detection → analyze in dependency order with injected import types
- **Loader:** `discover_imports()`, `extract_exports()`, `inject_import_types()`, `topological_sort()` — all implemented
- **Semantic:** `check_import()` and `check_from_import()` in `crates/nsl-semantic/src/checker.rs:558-598` — declares imported symbols in scope from `import_types` map
- **Semantic:** `analyze_with_imports()` public API that accepts `import_types: &HashMap<Symbol, Type>`
- **Codegen:** `compile_module()` (Linkage::Export, no main), `compile_entry()` (with imported_fns, structs, enums) in `crates/nsl-codegen/src/compiler.rs:1109-1168`
- **CLI:** `has_imports()` dispatch, `run_build_multi()` orchestration with temp .o files and multi-link in `crates/nsl-cli/src/main.rs:216-443`
- **Tests:** `examples/modules/main.nsl` + `utils.nsl` (relative import), `modules/math/helpers.nsl` + `main_subdir.nsl` (subdir import) — both working

**What's missing (what M13 adds):**

1. **Stdlib path resolution** — Resolver only handles relative paths. Need `NSL_STDLIB_PATH` env var and `<exe_dir>/stdlib/` fallback for `from nsl.math import clamp`
2. **Name mangling** — No mangling at all. Two modules defining a function with the same name → Cranelift linker crash. Need `nsl_math__clamp` mangling scheme
3. **`import X as Y` alias support in loader** — `discover_imports()` ignores `StmtKind::Import`. Need to resolve alias imports to files and pass their exports
4. **`import X as Y` semantic handling** — Need to support `math.clamp(...)` qualified access (dotted method calls resolving to module exports)
5. **Standard library files** — Need `stdlib/nsl/math.nsl` with math helpers
6. **`has_imports` update** — Only checks `from ... import`, not `import ... as`
7. **End-to-end tests** — Stdlib imports, alias imports, cross-module interactions

**What's deferred to later milestones:**
- `import nsl.math` without `as` (requires module objects at runtime)
- `__init__.nsl` package files
- `pub`/private visibility modifiers
- Incremental compilation / caching
- `@checkpoint`, `@backward`, `@custom_vjp` decorators
- `model.params()` intrinsic

---

### Task 1: Stdlib Path Resolution in Resolver

**Files:**
- Modify: `crates/nsl-cli/src/resolver.rs`

**Step 1: Write the failing test**

Create a manual test by trying to build a file that imports from `nsl.math` (which doesn't exist yet but would require stdlib resolution). This will fail because the resolver only looks relative to the importing file.

Run: `cargo build -p nsl-cli`
Expected: Compiles (no code changes yet, just baseline)

**Step 2: Implement stdlib path resolution**

Add a new function `resolve_stdlib_import()` and modify `resolve_import()` to try stdlib paths before relative resolution for paths starting with `nsl.`:

```rust
use std::env;

/// Resolve the stdlib root directory.
/// 1. Check NSL_STDLIB_PATH environment variable
/// 2. Fall back to <exe_dir>/stdlib/
fn stdlib_root() -> Option<PathBuf> {
    // Check env var first
    if let Ok(path) = env::var("NSL_STDLIB_PATH") {
        let p = PathBuf::from(path);
        if p.is_dir() {
            return Some(p);
        }
    }

    // Fall back to exe-relative stdlib/
    if let Ok(exe) = env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let stdlib = exe_dir.join("stdlib");
            if stdlib.is_dir() {
                return Some(stdlib);
            }
        }
    }

    None
}

pub fn resolve_import(
    import_path: &[Symbol],
    importing_file: &Path,
    interner: &Interner,
) -> Result<PathBuf, String> {
    if import_path.is_empty() {
        return Err("empty import path".to_string());
    }

    let segments: Vec<&str> = import_path
        .iter()
        .map(|sym| interner.resolve(sym.0).unwrap_or("<unknown>"))
        .collect();

    let module_name = segments.join(".");

    // Build relative path: nsl.math → nsl/math.nsl
    let mut rel = PathBuf::new();
    for (i, seg) in segments.iter().enumerate() {
        if i < segments.len() - 1 {
            rel.push(seg);
        } else {
            rel.push(format!("{seg}.nsl"));
        }
    }

    // 1. Try relative to importing file's directory (existing behavior)
    let base_dir = importing_file
        .parent()
        .ok_or_else(|| format!("cannot determine parent directory of '{}'", importing_file.display()))?;

    let candidate = base_dir.join(&rel);
    if candidate.is_file() {
        return candidate
            .canonicalize()
            .map_err(|e| format!("module '{module_name}' found at '{}' but cannot canonicalize: {e}", candidate.display()));
    }

    // 2. Try stdlib path (NSL_STDLIB_PATH or <exe_dir>/stdlib/)
    if let Some(stdlib) = stdlib_root() {
        let candidate = stdlib.join(&rel);
        if candidate.is_file() {
            return candidate
                .canonicalize()
                .map_err(|e| format!("module '{module_name}' found at '{}' but cannot canonicalize: {e}", candidate.display()));
        }
    }

    Err(format!(
        "module '{}' not found (searched: {}, stdlib)",
        module_name,
        base_dir.join(&rel).display()
    ))
}
```

**Step 3: Build and verify compilation**

Run: `cargo build -p nsl-cli`
Expected: PASS

**Step 4: Verify existing tests still pass**

Run: `cargo run -p nsl-cli -- run examples/modules/main.nsl`
Expected: Output matches `tests/expected/modules_main.txt` (7, Hello world!, 42)

**Step 5: Commit**

```bash
git add crates/nsl-cli/src/resolver.rs
git commit -m "feat(m13): add stdlib path resolution (NSL_STDLIB_PATH + exe-relative fallback)"
```

---

### Task 2: Name Mangling Infrastructure

**Files:**
- Create: `crates/nsl-cli/src/mangling.rs`
- Modify: `crates/nsl-cli/src/main.rs` (add `mod mangling;`)

**Step 1: Create the mangling module**

```rust
// crates/nsl-cli/src/mangling.rs

use std::path::Path;

/// Compute a module prefix from a file path relative to a base directory.
///
/// Example: base="/project", path="/project/nsl/math.nsl" → "nsl_math"
/// Example: base="/project", path="/project/utils.nsl" → "utils"
pub fn module_prefix(path: &Path, base_dir: &Path) -> String {
    // Get path relative to base
    let rel = path.strip_prefix(base_dir).unwrap_or(path);

    // Remove .nsl extension
    let stem = rel.with_extension("");

    // Convert separators to underscores
    stem.to_string_lossy()
        .replace(['/', '\\'], "_")
}

/// Mangle a function name with its module prefix.
///
/// Example: prefix="nsl_math", name="clamp" → "nsl_math__clamp"
/// Entry module functions are NOT mangled (prefix is empty).
pub fn mangle(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}__{name}")
    }
}

/// Demangle a symbol name for error messages.
///
/// Example: "nsl_math__clamp" → "nsl.math.clamp"
pub fn demangle(mangled: &str) -> String {
    if let Some((prefix, name)) = mangled.split_once("__") {
        let module = prefix.replace('_', ".");
        format!("{module}.{name}")
    } else {
        mangled.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_module_prefix() {
        let base = PathBuf::from("/project");
        assert_eq!(module_prefix(Path::new("/project/nsl/math.nsl"), &base), "nsl_math");
        assert_eq!(module_prefix(Path::new("/project/utils.nsl"), &base), "utils");
    }

    #[test]
    fn test_mangle() {
        assert_eq!(mangle("nsl_math", "clamp"), "nsl_math__clamp");
        assert_eq!(mangle("", "main_helper"), "main_helper");
    }

    #[test]
    fn test_demangle() {
        assert_eq!(demangle("nsl_math__clamp"), "nsl.math.clamp");
        assert_eq!(demangle("main_helper"), "main_helper");
    }
}
```

**Step 2: Register the module in main.rs**

Add `mod mangling;` to `crates/nsl-cli/src/main.rs` after the existing `mod` declarations.

**Step 3: Run the unit tests**

Run: `cargo test -p nsl-cli`
Expected: All 3 tests pass

**Step 4: Commit**

```bash
git add crates/nsl-cli/src/mangling.rs crates/nsl-cli/src/main.rs
git commit -m "feat(m13): add name mangling module for Cranelift symbol uniqueness"
```

---

### Task 3: Integrate Name Mangling into Multi-File Build

**Files:**
- Modify: `crates/nsl-cli/src/main.rs` — `run_build_multi()` function
- Modify: `crates/nsl-cli/src/loader.rs` — add `module_prefix` field to `ModuleData`

**Step 1: Add module_prefix to ModuleData**

In `crates/nsl-cli/src/loader.rs`, add to `ModuleData`:
```rust
/// Mangling prefix for this module's symbols (e.g., "nsl_math")
pub module_prefix: String,
```

Compute it in `load_all_modules()` when building `ModuleData`. The base directory is the entry file's parent directory (for relative imports) or the stdlib root (for stdlib imports).

For simplicity, use the module's path relative to the entry file's parent:
```rust
let base_dir = entry_path.parent().unwrap_or(Path::new("."));
let prefix = if *path == entry_path {
    String::new() // entry module — no prefix
} else {
    crate::mangling::module_prefix(path, base_dir)
};
```

**Step 2: Use mangled names in run_build_multi's entry codegen**

In `run_build_multi()` in `main.rs`, when building `imported_fns` for `compile_entry()`, mangle the function names:

Change:
```rust
imported_fns.push((name, sig));
```
To:
```rust
let mangled = crate::mangling::mangle(&dep_data.module_prefix, &name);
imported_fns.push((mangled, sig));
```

**Step 3: Use mangled names in compile_module**

In `run_build_multi()`, when compiling non-entry modules, pass the module prefix so that `compile_module` uses mangled function names. This requires adding a `module_prefix` parameter to `compile_module()`.

In `crates/nsl-codegen/src/compiler.rs`, modify `compile_module`:
```rust
pub fn compile_module(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    module_prefix: &str,
    dump_ir: bool,
) -> Result<Vec<u8>, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map)?;
    compiler.dump_ir = dump_ir;
    compiler.module_prefix = module_prefix.to_string();
    // ... rest unchanged
```

Add `module_prefix: String` field to the `Compiler` struct (default empty string).

In `declare_user_functions_with_linkage()`, when declaring function names, mangle them:
```rust
let raw_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>");
let func_name = if self.module_prefix.is_empty() {
    raw_name.to_string()
} else {
    crate::mangling::mangle(&self.module_prefix, raw_name) // Note: need to re-export or inline
};
```

Wait — `nsl-codegen` doesn't depend on `nsl-cli` (it's the other way around). The mangling logic must either:
- Be duplicated as a simple inline function in codegen, OR
- Be moved to a shared crate (like `nsl-ast` or a new `nsl-common`)

**Simplest approach:** Add `mangle()` as a simple utility function directly in `nsl-codegen/src/compiler.rs`:

```rust
/// Mangle a function name with its module prefix for Cranelift symbol uniqueness.
fn mangle_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}__{name}")
    }
}
```

And keep the `nsl-cli/src/mangling.rs` for the module_prefix computation + demangle + tests.

**Step 4: Update compile_module call site in main.rs**

```rust
match nsl_codegen::compile_module(
    &mod_data.ast,
    &interner,
    &mod_data.type_map,
    &mod_data.module_prefix,  // NEW
    dump_ir,
)
```

**Step 5: Update the public API**

In `crates/nsl-codegen/src/lib.rs`, update the `compile_module` re-export signature.

**Step 6: Build and verify**

Run: `cargo build -p nsl-cli`
Expected: PASS

**Step 7: Run existing multi-module test**

Run: `cargo run -p nsl-cli -- run examples/modules/main.nsl`
Expected: Same output (7, Hello world!, 42)

Note: This test uses relative imports (`from utils import add`), so the mangling will change the Cranelift symbol names (`utils__add` etc.) but the entry module's import declarations will also use the mangled names, so linking still works.

**Step 8: Commit**

```bash
git add crates/nsl-cli/src/loader.rs crates/nsl-cli/src/main.rs crates/nsl-codegen/src/compiler.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(m13): integrate name mangling into multi-file compilation pipeline"
```

---

### Task 4: `import X as Y` Support in Loader

**Files:**
- Modify: `crates/nsl-cli/src/loader.rs` — `discover_imports()` function
- Modify: `crates/nsl-ast/src/decl.rs` — add `alias` field to `ImportStmt` (if not present)

**Step 1: Check ImportStmt for alias field**

The current `ImportStmt` has `path` and `items` but no `alias` field. The parser's `parse_import_stmt()` doesn't handle `as`. We need:

1. Add `alias: Option<Symbol>` to `ImportStmt` in AST
2. Parse `import nsl.math as math` in the parser
3. Handle alias imports in the loader's `discover_imports()`

**Step 2: Add alias to ImportStmt AST**

In `crates/nsl-ast/src/decl.rs`, modify:
```rust
pub struct ImportStmt {
    pub path: Vec<Symbol>,
    pub items: ImportItems,
    pub alias: Option<Symbol>,
    pub span: Span,
}
```

**Step 3: Parse `import X as Y` in parser**

In `crates/nsl-parser/src/decl.rs`, in `parse_import_stmt()`, after collecting all path segments and before `p.expect_end_of_stmt()`, add alias parsing:

```rust
// Check for alias: import nsl.math as math
let alias = if p.eat(&TokenKind::As) {
    let (a, _) = p.expect_ident();
    Some(a)
} else {
    None
};

p.expect_end_of_stmt();
Stmt {
    kind: StmtKind::Import(ImportStmt {
        path,
        items: ImportItems::Module,
        alias,
        span: start.merge(p.prev_span()),
    }),
    ...
}
```

Also update the other `ImportStmt` construction sites in the same function (for Glob and Named variants) to include `alias: None`.

**Step 4: Update discover_imports to handle Import statements**

In `crates/nsl-cli/src/loader.rs`, `discover_imports()`:

```rust
for stmt in stmts {
    match &stmt.kind {
        StmtKind::FromImport(from_import) => {
            let resolved = resolver::resolve_import(
                &from_import.module_path,
                source_file,
                interner,
            )?;
            imports.push((resolved, ImportInfo::From(from_import.clone())));
        }
        StmtKind::Import(import_stmt) => {
            if import_stmt.alias.is_some() {
                // `import nsl.math as math` — resolve the module
                let resolved = resolver::resolve_import(
                    &import_stmt.path,
                    source_file,
                    interner,
                )?;
                imports.push((resolved, ImportInfo::Alias(import_stmt.clone())));
            }
            // Plain `import nsl.math` without alias — skip for now (not supported)
        }
        _ => {}
    }
}
```

This requires changing the return type from `Vec<(PathBuf, FromImportStmt)>` to `Vec<(PathBuf, ImportInfo)>` where:

```rust
enum ImportInfo {
    From(FromImportStmt),
    Alias(ImportStmt),
}
```

**Step 5: Update load_all_modules to handle ImportInfo::Alias**

In the Phase 3 analysis loop, when building `import_types` for a module, handle alias imports by injecting all exports from the dependency under the alias. This requires the semantic checker to understand `math.clamp(...)` as a qualified name.

For the simplest approach: when we encounter `import nsl.math as math`, inject all exports from `nsl.math` into `import_types` with a prefix notation. However, this is complex because the semantic checker uses `Symbol` keys.

**Simpler approach:** For `import X as Y`, treat it like `from X import *` but prefix with alias. The semantic checker already handles dotted access on identifiers through `FieldAccess`. We need to:

1. Declare alias `math` as a special "module" type in the semantic scope
2. When the checker sees `math.clamp(...)`, resolve it via the module's exports

This is more involved. Let's defer the semantic part to Task 5 and focus on getting the parser + loader changes compiling here.

**Step 6: Update ModuleData dependencies**

Change `ModuleData.dependencies` from `Vec<PathBuf>` to store import info alongside paths. Or simpler: just keep `Vec<PathBuf>` for dependencies and add a separate field for alias mappings.

Add to `ModuleData`:
```rust
/// Alias imports: alias_symbol → module_path (for `import X as Y`)
pub alias_imports: Vec<(Symbol, PathBuf)>,
```

**Step 7: Build and verify**

Run: `cargo build -p nsl-cli`
Expected: PASS

**Step 8: Commit**

```bash
git add crates/nsl-ast/src/decl.rs crates/nsl-parser/src/decl.rs crates/nsl-cli/src/loader.rs
git commit -m "feat(m13): add import-as alias parsing and loader support"
```

---

### Task 5: Semantic Support for `import X as Y` (Qualified Access)

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs` — `check_import()` and expression checking
- Modify: `crates/nsl-semantic/src/types.rs` — add `Type::Module` variant

**Step 1: Add Module type**

In `crates/nsl-semantic/src/types.rs`, add:
```rust
Module {
    exports: HashMap<Symbol, Box<Type>>,
},
```

(Use `Box<Type>` to avoid recursive type size issues.)

**Step 2: Update check_import for alias imports**

In `check_import()`, when `import.alias.is_some()` and `import.items == ImportItems::Module`:

```rust
ImportItems::Module => {
    if let Some(alias) = import.alias {
        // `import nsl.math as math` — declare alias as Module type
        // The loader has populated import_types with all exports under special keys
        // Build a Module type from the imports
        let ty = Type::Module { exports: /* collected from import_types */ };
        self.declare_symbol(alias, ty, import.span, true, false);
    } else {
        // existing behavior: declare last path segment
        if let Some(last) = import.path.last() {
            let ty = self.import_types.get(last).cloned().unwrap_or(Type::Unknown);
            self.declare_symbol(*last, ty, import.span, true, false);
        }
    }
}
```

The challenge: how does the checker know which exports belong to the alias import? The loader needs to pass this information.

**Best approach:** In `loader.rs`, when injecting import types for an alias import, store them under a convention. For example, for `import nsl.math as math`, store:
- `import_types[math] = Type::Module { exports: {clamp: fn_type, lerp: fn_type, ...} }`

This means in `inject_import_types()` (or a new `inject_alias_import()`), we build the Module type directly:

```rust
fn inject_alias_import(
    alias: Symbol,
    dep_exports: &HashMap<Symbol, Type>,
    import_types: &mut HashMap<Symbol, Type>,
) {
    let exports: HashMap<Symbol, Box<Type>> = dep_exports
        .iter()
        .map(|(sym, ty)| (*sym, Box::new(ty.clone())))
        .collect();
    import_types.insert(alias, Type::Module { exports });
}
```

**Step 3: Handle qualified access in expression checking**

In `check_expr()`, the `ExprKind::FieldAccess` or `ExprKind::MethodCall` case: when the receiver is an identifier that resolves to `Type::Module`, look up the field/method in the module's exports.

For `math.clamp(x, lo, hi)`:
- `math` resolves to `Type::Module { exports: {clamp: FnType(...)} }`
- `.clamp` is a field access on a module → returns the function type
- `(x, lo, hi)` is a call on that function type

This should work with existing `FieldAccess` checking if we add a case for `Type::Module`.

**Step 4: Build and verify**

Run: `cargo build -p nsl-cli`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/nsl-semantic/src/types.rs crates/nsl-semantic/src/checker.rs crates/nsl-cli/src/loader.rs
git commit -m "feat(m13): add Type::Module and qualified access for import-as aliases"
```

---

### Task 6: Codegen Support for Alias Import Calls

**Files:**
- Modify: `crates/nsl-codegen/src/expr.rs` or `crates/nsl-codegen/src/call.rs` — handle `module.func(...)` calls
- Modify: `crates/nsl-cli/src/main.rs` — pass alias-to-mangled-name mapping to compile_entry

**Step 1: Pass alias mapping to codegen**

When calling `compile_entry()` from `run_build_multi()`, we need the codegen to know that `math.clamp(...)` should emit a call to `nsl_math__clamp`.

Add a new parameter to `compile_entry()`:
```rust
module_aliases: &HashMap<String, Vec<(String, String)>>,
// alias_name → [(local_fn_name, mangled_fn_name)]
```

Or simpler: build a flat map of `(alias, fn_name) → mangled_name` and pass it in.

**Step 2: Handle qualified calls in codegen**

In the codegen's call expression handling, when encountering a `FieldAccess` call like `math.clamp(x, y, z)`:

1. Check if the receiver is a module alias
2. Look up the mangled name for `math` + `clamp`
3. Emit a call to the mangled Cranelift function

This ties into how the compiler resolves function calls. The `compile_call()` or `compile_call_expr()` function needs to check for qualified module calls.

**Step 3: Build and verify**

Run: `cargo build -p nsl-cli`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/nsl-codegen/src/expr.rs crates/nsl-codegen/src/compiler.rs crates/nsl-cli/src/main.rs
git commit -m "feat(m13): codegen support for qualified module.func() calls"
```

---

### Task 7: Update `has_imports()` Detection

**Files:**
- Modify: `crates/nsl-cli/src/main.rs` — `has_imports()` function

**Step 1: Update the import detection**

Current code only checks for `from ... import`. Also check for `import ... as`:

```rust
fn has_imports(file: &PathBuf) -> bool {
    if let Ok(source) = std::fs::read_to_string(file) {
        source.lines().any(|line| {
            let trimmed = line.trim();
            (trimmed.starts_with("from ") && trimmed.contains(" import "))
                || (trimmed.starts_with("import ") && trimmed.contains(" as "))
        })
    } else {
        false
    }
}
```

**Step 2: Build and verify**

Run: `cargo build -p nsl-cli`
Expected: PASS

**Step 3: Commit**

```bash
git add crates/nsl-cli/src/main.rs
git commit -m "feat(m13): detect import-as statements in has_imports()"
```

---

### Task 8: Create Standard Library — `stdlib/nsl/math.nsl`

**Files:**
- Create: `stdlib/nsl/math.nsl`

**Step 1: Create stdlib directory structure**

```bash
mkdir -p stdlib/nsl
```

**Step 2: Write math.nsl**

```nsl
# NSL Standard Library — nsl.math
# Basic math utility functions

fn clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

fn lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

fn sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0

fn max_val(a: float, b: float) -> float:
    if a > b:
        return a
    return b

fn min_val(a: float, b: float) -> float:
    if a < b:
        return a
    return b
```

**Step 3: Verify it parses**

Run: `cargo run -p nsl-cli -- check stdlib/nsl/math.nsl`
Expected: "OK: stdlib/nsl/math.nsl checked successfully (5 statements)"

**Step 4: Commit**

```bash
git add stdlib/nsl/math.nsl
git commit -m "feat(m13): add stdlib nsl.math module with clamp, lerp, sign, max_val, min_val"
```

---

### Task 9: End-to-End Test — Stdlib `from` Import

**Files:**
- Create: `examples/m13_stdlib_import.nsl`
- Create: `tests/expected/m13_stdlib_import.txt`

**Step 1: Write the test program**

```nsl
# M13 Stdlib Import Test
from nsl.math import clamp, lerp, sign

print(clamp(5.0, 0.0, 3.0))
print(clamp(-1.0, 0.0, 10.0))
print(lerp(0.0, 10.0, 0.5))
print(sign(-42.0))
print(sign(0.0))
print(sign(7.0))
```

**Step 2: Write expected output**

```
3.0
0.0
5.0
-1.0
0.0
1.0
```

**Step 3: Run the test**

Run: `NSL_STDLIB_PATH=stdlib cargo run -p nsl-cli -- run examples/m13_stdlib_import.nsl`

Expected: Output matches expected file exactly.

If it fails, debug: check resolver finds `stdlib/nsl/math.nsl`, check mangling consistency between `compile_module` and `compile_entry`, check linker resolves symbols.

**Step 4: Commit**

```bash
git add examples/m13_stdlib_import.nsl tests/expected/m13_stdlib_import.txt
git commit -m "test(m13): add stdlib from-import end-to-end test"
```

---

### Task 10: End-to-End Test — `import X as Y` Alias

**Files:**
- Create: `examples/m13_import_alias.nsl`
- Create: `tests/expected/m13_import_alias.txt`

**Step 1: Write the test program**

```nsl
# M13 Import Alias Test
import nsl.math as math

print(math.clamp(100.0, 0.0, 50.0))
print(math.sign(-3.14))
print(math.lerp(0.0, 100.0, 0.25))
```

**Step 2: Write expected output**

```
50.0
-1.0
25.0
```

**Step 3: Run the test**

Run: `NSL_STDLIB_PATH=stdlib cargo run -p nsl-cli -- run examples/m13_import_alias.nsl`

Expected: Output matches expected file.

**Step 4: Commit**

```bash
git add examples/m13_import_alias.nsl tests/expected/m13_import_alias.txt
git commit -m "test(m13): add import-as alias end-to-end test"
```

---

### Task 11: End-to-End Test — Mixed Imports + Relative + Stdlib

**Files:**
- Create: `examples/m13_mixed_imports.nsl`
- Create: `examples/m13_helpers.nsl`
- Create: `tests/expected/m13_mixed_imports.txt`

**Step 1: Write the helper module**

`examples/m13_helpers.nsl`:
```nsl
fn double(x: float) -> float:
    return x * 2.0

fn negate(x: float) -> float:
    return 0.0 - x
```

**Step 2: Write the main test program**

`examples/m13_mixed_imports.nsl`:
```nsl
# M13 Mixed Import Test — stdlib + relative in same file
from nsl.math import clamp
from m13_helpers import double, negate

let x = double(3.0)
print(x)
print(clamp(x, 0.0, 5.0))
print(negate(x))
```

**Step 3: Write expected output**

```
6.0
5.0
-6.0
```

**Step 4: Run the test**

Run: `NSL_STDLIB_PATH=stdlib cargo run -p nsl-cli -- run examples/m13_mixed_imports.nsl`

Expected: Output matches expected file.

**Step 5: Commit**

```bash
git add examples/m13_mixed_imports.nsl examples/m13_helpers.nsl tests/expected/m13_mixed_imports.txt
git commit -m "test(m13): add mixed stdlib + relative import end-to-end test"
```

---

### Task 12: Verify All Existing Tests Pass

**Files:** None (verification only)

**Step 1: Run all existing example tests**

Run each example program and verify output matches expected:

```bash
cargo run -p nsl-cli -- run examples/hello.nsl
cargo run -p nsl-cli -- run examples/features.nsl
cargo run -p nsl-cli -- run examples/m5_features.nsl
cargo run -p nsl-cli -- run examples/m6_features.nsl
cargo run -p nsl-cli -- run examples/m8_features.nsl
cargo run -p nsl-cli -- run examples/m9_tensors.nsl
cargo run -p nsl-cli -- run examples/m10_shape_check.nsl
cargo run -p nsl-cli -- run examples/m11_model_basic.nsl
cargo run -p nsl-cli -- run examples/m11_model_tensor.nsl
cargo run -p nsl-cli -- run examples/m12_grad_basic.nsl
cargo run -p nsl-cli -- run examples/m12_grad_matmul.nsl
cargo run -p nsl-cli -- run examples/m12_grad_model.nsl
cargo run -p nsl-cli -- run examples/m12_no_grad.nsl
cargo run -p nsl-cli -- run examples/modules/main.nsl
```

Compare each with its expected file in `tests/expected/`.

**Step 2: Run M13-specific tests**

```bash
NSL_STDLIB_PATH=stdlib cargo run -p nsl-cli -- run examples/m13_stdlib_import.nsl
NSL_STDLIB_PATH=stdlib cargo run -p nsl-cli -- run examples/m13_import_alias.nsl
NSL_STDLIB_PATH=stdlib cargo run -p nsl-cli -- run examples/m13_mixed_imports.nsl
```

**Step 3: Run cargo unit tests**

Run: `cargo test --workspace`
Expected: All tests pass (including mangling unit tests from Task 2)

**Step 4: Run clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings

**Step 5: Commit (if any fixups needed)**

```bash
git commit -m "chore(m13): fix any issues found during full test suite verification"
```

---

## Summary

| Task | Description | Key Files |
|------|------------|-----------|
| 1 | Stdlib path resolution | `resolver.rs` |
| 2 | Name mangling module | `mangling.rs` (new) |
| 3 | Integrate mangling into build | `loader.rs`, `main.rs`, `compiler.rs` |
| 4 | `import X as Y` parser + loader | `decl.rs` (ast + parser), `loader.rs` |
| 5 | `import X as Y` semantic support | `checker.rs`, `types.rs`, `loader.rs` |
| 6 | `import X as Y` codegen | `expr.rs`, `compiler.rs`, `main.rs` |
| 7 | `has_imports()` update | `main.rs` |
| 8 | Stdlib `nsl.math` | `stdlib/nsl/math.nsl` (new) |
| 9 | Test: stdlib from-import | `m13_stdlib_import.nsl` |
| 10 | Test: import alias | `m13_import_alias.nsl` |
| 11 | Test: mixed imports | `m13_mixed_imports.nsl` |
| 12 | Full regression suite | (verification only) |

**Critical path:** Tasks 1-3 (stdlib resolution + mangling) → Task 8 (stdlib file) → Task 9 (first end-to-end test). Tasks 4-6 (alias support) can be done after 9 passes. Task 7 is small and can slot in anywhere. Tasks 10-12 are verification.
