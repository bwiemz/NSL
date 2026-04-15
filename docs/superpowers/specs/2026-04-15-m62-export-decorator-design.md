# M62 `@export` Decorator — Design

**Date:** 2026-04-15
**Status:** Approved for implementation
**Branch (target):** `feat/m62-finish`
**Predecessor:** [M62 Legacy Interop spec](2026-03-19-m62-legacy-interop-design.md) §1 Language Surface, §7 Codegen Changes, §11 Testing Strategy — partial implementation audit in the 2026-04-15 session identified `@export` as the missing gate on the shared-lib path.

## 1. Goal

Make `nsl build --shared-lib model.nsl` produce a `.so`/`.dylib`/`.dll` with C-callable symbols. Today, all user functions get `Linkage::Local` unconditionally at [declaration.rs:29](../../../crates/nsl-codegen/src/compiler/declaration.rs#L29) regardless of the `--shared-lib` flag, so the compiled library has zero reachable entry points. This spec adds the `@export` decorator that opts specific functions into `Linkage::Export`, names the exported symbol unmangled (or via `name="..."`), and emits a matching C header.

Policy: strict opt-in — **only `@export` functions become C-callable**. Everything else stays `Linkage::Local`. This preserves ABI stability (internal refactors don't break downstream binaries) and keeps symbol-table noise low (no stdlib/helper leakage).

## 2. Non-Goals (Explicitly Deferred)

- `--export=fn1,fn2,...` CLI fallback flag. Source-level decoration is more discoverable.
- Python type stub generation (`.pyi`).
- Hot-reload of `@export` signatures — recompile required.
- Auto-warning when `--shared-lib` yields zero `@export` functions (revisit if users ask).
- Multi-module shared libs with `@export` distributed across compile units. Each compile unit stays independent.
- Fixing the grad-context bridge bugs (Bug 1/2/3 from the audit). Separate spec.

## 3. Language Surface

No parser or AST changes. The generic decorator infrastructure at [parser/stmt.rs:428](../../../crates/nsl-parser/src/stmt.rs#L428) already produces `Decorator { name: Vec<Symbol>, args: Option<Vec<Arg>>, span }` for `@export` and `@export(name="...")`.

```
@export
fn forward(input: Tensor<[B, S], int32>) -> Tensor<[B, S, V], f32>:
    let hidden = model.embed(input)
    return model.decode(hidden)

@export(name="predict")
fn inference_forward(x: Tensor<[B, D], f32>) -> Tensor<[B, C], f32>:
    return model(x)
```

Keyword `name` is the only recognized argument. Other kwargs are a compile error (Task B).

## 4. Codegen

### 4.1 Linkage and symbol naming in `declaration.rs`

Extend the existing decorator loop at [declaration.rs:62-76](../../../crates/nsl-codegen/src/compiler/declaration.rs#L62) to recognize `@export` BEFORE the `declare_function` call at line 54:

```rust
// [in declare_user_functions_with_linkage, per-function loop]
let raw_name = self.resolve_sym(fn_def.name).to_string();
let cranelift_name = mangle_name(&self.module_prefix, &raw_name);
let sig = self.build_fn_signature(fn_def);

let mut effective_linkage = linkage;          // from caller (Local by default)
let mut export_override: Option<String> = None;
let mut is_export = false;

if let Some(decos) = decorators {
    for d in decos {
        if d.name.len() == 1 && self.resolve_sym(d.name[0]) == "export" {
            effective_linkage = Linkage::Export;
            is_export = true;
            if let Some(ref args) = d.args {
                for arg in args {
                    if arg.name.map(|s| self.resolve_sym(s).to_string()) == Some("name".to_string()) {
                        if let ExprKind::StringLiteral(s) = &arg.value.kind {
                            export_override = Some(s.clone());
                        }
                    }
                }
            }
        }
    }
}

// Exported symbols skip module-prefix mangling so C consumers see the
// unmangled (or user-chosen) name.
let symbol_name = if is_export {
    export_override.clone().unwrap_or_else(|| raw_name.clone())
} else {
    cranelift_name.clone()
};

let func_id = self.module.declare_function(&symbol_name, effective_linkage, &sig)?;
self.registry.functions.insert(raw_name.clone(), (func_id, sig.clone()));

// Track for header emission (Task C)
if is_export {
    self.features.export_functions.push(ExportInfo {
        symbol_name: symbol_name.clone(),
        raw_name: raw_name.clone(),
        params: /* extract from fn_def.params via type check */,
        return_type: /* extract from fn_def.return_type */,
    });
}
```

The existing `@no_grad` / `@test` / `@fp8_compute` / `@grammar` branches stay intact inside their existing loop; the `@export` branch runs before them so linkage/symbol decisions are locked in first.

### 4.2 `ExportInfo` data model

File: `crates/nsl-codegen/src/compiler/mod.rs` — added as a field on `Compiler.features`:

```rust
pub struct ExportInfo {
    pub symbol_name: String,              // appears in the .so's export table
    pub raw_name: String,                 // NSL-side function name
    pub params: Vec<ExportParamInfo>,
    pub return_type: ExportTypeInfo,
}

pub struct ExportParamInfo {
    pub name: String,                     // NSL parameter name
    pub ty: ExportTypeInfo,
}

pub enum ExportTypeInfo {
    Tensor {
        // Shape dims as strings — named dims (`"B"`) stay named, literals (`4`) stringify to `"4"`.
        shape: Vec<String>,
        dtype: ExportDtype,
        device: ExportDevice,
    },
    Scalar(ExportDtype),                  // i32, i64, f32, f64, bool
    Tuple(Vec<ExportTypeInfo>),           // multi-output functions
}

pub enum ExportDtype {
    F32, F64, F16, BF16, I8, I16, I32, I64, U8, U16, U32, U64, Bool,
}

pub enum ExportDevice {
    Cpu,
    Cuda,
    Any,                                  // compiler chooses at call time
}

impl Compiler<'_> {
    pub fn new_with_...(...) -> Self {
        Self {
            /* existing fields */
            features: CompilerFeatures {
                /* existing */
                export_functions: Vec::new(),
            },
        }
    }
}
```

`ExportInfo` is constructed from the type-checked `fn_def` signature. Type lowering logic lives in a new `crates/nsl-codegen/src/c_header.rs` helper that both the codegen-side tracking (Task A) and header emission (Task C) call.

## 5. Semantic Validation

File: `crates/nsl-semantic/src/export.rs` (new). Invoked during the existing decorator-aware semantic pass (see [checker/stmt.rs:237, 329, 447, 523, 664](../../../crates/nsl-semantic/src/checker/stmt.rs) for the `@freeze` / `@checkpoint` / `@autotune` precedent).

### 5.1 Checks

| Constraint | Error message |
|---|---|
| `@export` on non-function (model, let, expr, type alias) | `@export can only be applied to functions` |
| `@export` on function with closure parameter (`\|x\| -> ...` or `fn(T) -> U` arg type) | `@export function cannot take closure parameters — closures cannot cross the C ABI` |
| `@export` on function returning non-tensor/scalar/tuple-of-those (struct, model, closure, unit) | `@export function must return a tensor, scalar, or tuple of those; got <type>` |
| `@export` on function with generic type params (`fn foo<T>(...)`) | `@export function cannot have generic type parameters — C ABI requires monomorphized types` |
| `@export(name="")` | `@export(name=\"...\") cannot be empty` |
| `@export(name="<non-C-identifier>")` — matches `^[A-Za-z_][A-Za-z0-9_]*$` | `@export(name=\"<value>\") must be a valid C identifier: letters, digits, underscore; cannot start with digit` |
| `@export(x=..., y=...)` — any kwarg other than `name` | `@export takes only a 'name' keyword argument; got '<other>'` |
| `@export(positional)` — positional args | `@export takes only a 'name=...' keyword argument; no positional arguments` |
| `@export` appearing twice on the same function | `@export decorator appears multiple times on 'foo'` |

### 5.2 Module scope

The validation module exposes a single entry:

```rust
pub fn validate_exports(module: &Module, interner: &Interner) -> Vec<SemanticError>
```

Called from the existing semantic pass that already walks top-level declarations. Errors collect into the same `Vec<SemanticError>` that other validations populate; reporting happens through the existing diagnostic renderer.

## 6. C Header Emission

File: `crates/nsl-codegen/src/c_header.rs` (new). Emits `<output>.h` alongside the shared library when `--shared-lib` is set AND `Compiler.features.export_functions` is non-empty.

### 6.1 Trigger

CLI handler at `crates/nsl-cli/src/main.rs`, after the shared-library build completes. Looks up `Compiler.features.export_functions`, calls `c_header::emit(path, &exports) -> Result<(), CodegenError>`, writes sibling `.h` file.

Header path: same stem as the shared library, `.h` extension. Example: `nsl build --shared-lib model.nsl -o libmodel.so` produces `libmodel.so` + `libmodel.h`.

### 6.2 Output shape

```c
/* model.h — Auto-generated by NSL compiler */
#ifndef NSL_MODEL_H
#define NSL_MODEL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct NslModel NslModel;

/* Tensor descriptor for the C API */
typedef struct {
    void*    data;
    int64_t* shape;
    int64_t* strides;      /* NULL = contiguous */
    int32_t  ndim;
    int32_t  dtype;        /* 0=f32, 1=f64, 2=f16, 3=bf16, 4=i32, 5=i64, 6=i8, 7=u8 */
    int32_t  device_type;  /* 0=CPU, 1=CUDA */
    int32_t  device_id;
} NslTensorDesc;

/* Lifecycle (provided by libnsl_runtime, declared here for convenience) */
NslModel* nsl_model_create(const char* weights_path);
void      nsl_model_destroy(NslModel* model);

/* @export functions */
int forward(NslModel* model,
            const NslTensorDesc* input,
            NslTensorDesc* __ret);

int predict(NslModel* model,
            const NslTensorDesc* x,
            NslTensorDesc* __ret);

#ifdef __cplusplus
}
#endif
#endif /* NSL_MODEL_H */
```

### 6.3 Type lowering rules

| NSL type | C type in header |
|---|---|
| `Tensor<[...], dtype, device>` (input) | `const NslTensorDesc* <param_name>` |
| `Tensor<[...], dtype, device>` (return) | `NslTensorDesc* __ret` (caller-allocated) |
| `Tuple<(Tensor<...>, Tensor<...>)>` (return) | `NslTensorDesc* __rets` (caller-allocated array of N) — preceded by an `int __num_rets` out-param |
| `int` / `i32` (input) | `int32_t` |
| `int` / `i64` (input) | `int64_t` |
| `float` / `f32` (input) | `float` |
| `float` / `f64` (input) | `double` |
| `bool` (input) | `int32_t` (0 or 1) |

Return codes: all `@export` functions return `int` (0 = success, non-zero = error code matching the runtime's error-code convention used by existing `nsl_model_*` functions).

### 6.4 Module handle threading

Every `@export` function's first parameter is an implicit `NslModel* model` in the C header — the handle that `nsl_model_create` returns. On the NSL side, the function doesn't declare this parameter; codegen synthesizes it when wrapping the NSL function for C ABI (the NSL function body reads model weights through the usual mechanism; the C-level wrapper stores/retrieves the handle via the runtime's thread-local).

This matches how the existing `nsl_model_forward` in [c_api.rs:255](../../../crates/nsl-runtime/src/c_api.rs#L255) threads the handle. `@export` functions get the same treatment so they compose naturally with the runtime's lifecycle.

## 7. No CLI Changes

Skipping `--export=fn1,fn2,...`. Users choose which functions to export by editing source. Revisit if there's demand.

## 8. Testing

### 8.1 Codegen unit tests (`crates/nsl-codegen/src/compiler/declaration.rs` test module)

1. `@export fn foo` → declared function has `Linkage::Export`, symbol name `"foo"` (not `module_foo`).
2. `@export(name="bar") fn foo` → symbol name `"bar"`, linkage `Export`.
3. `fn foo` (undecorated) in a shared-lib build → symbol name `module_foo` (mangled), linkage `Local`.
4. `@no_grad @export fn foo` → `@no_grad` registered in `no_grad_fns` AND linkage `Export` — decorators compose.

### 8.2 Semantic validation tests (`crates/nsl-semantic/src/export.rs` test module)

5. `@export model Foo { ... }` → error: `@export can only be applied to functions`.
6. `@export fn foo(cb: fn(Tensor) -> Tensor)` → error: closure params.
7. `@export fn foo() -> Bar` (where `Bar` is a struct) → error: non-tensor return.
8. `@export fn foo<T>(x: Tensor<[4], T>)` → error: generic type params.
9. `@export(name="")` → error: empty name.
10. `@export(name="123bad")` → error: invalid C identifier.
11. `@export(name="ok", other=1)` → error: unknown kwarg.
12. `@export("positional")` → error: positional args.
13. `@export @export fn foo` → error: duplicate decorator.

### 8.3 Header-emission snapshot test

14. `crates/nsl-codegen/tests/c_header_snapshot.rs` — compile a fixture with two `@export` functions (one default-named, one `name=`-overridden) and snapshot the generated header bytes. String-match on function prototypes.

### 8.4 E2E test

15. `examples/m62_shared_lib.nsl` — smallest fixture demonstrating the full loop:

```
@export
fn add(a: Tensor<[4], f32>, b: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return a + b
```

Test at `python/tests/test_m62_export.py`:
- `nsl build --shared-lib examples/m62_shared_lib.nsl -o /tmp/libadd.so`
- Load with `ctypes.CDLL("/tmp/libadd.so")` — assert `add` symbol exists.
- Call `add` with two `NslTensorDesc` inputs via the existing `_bridge.py` helpers, verify output tensor equals `a + b` element-wise.
- Assert the generated `libadd.h` contains `int add(NslModel*, const NslTensorDesc*, const NslTensorDesc*, NslTensorDesc*);`.

### 8.5 Regression guard

All existing non-M62 tests continue to pass unchanged. No existing test uses `@export`; all functions continue to be Local → emitted IR is byte-identical (the new decorator scan is a no-op when `@export` isn't present).

## 9. Architecture Diagram

```
NSL source                         Parser                   AST
@export fn forward(x)         →    parse_decorator    →    StmtKind::Decorated {
                                   (already exists)          decorators: [Decorator{name:["export"]}],
                                                             stmt: FnDef { name: "forward", ... },
                                                         }

                                          │
                                          ▼
                              Semantic: validate_exports
                              (new — crates/nsl-semantic/src/export.rs)
                                          │
                                          ▼
                              Codegen: declaration.rs decorator loop
                              @export branch (new):
                                ├─ linkage = Linkage::Export
                                ├─ symbol = name="..." ?? raw_name
                                └─ push ExportInfo onto
                                   self.features.export_functions
                                          │
                                          ▼
                              module.declare_function(symbol, Export, sig)
                                          │
                                          ▼
                              libmodel.so (symbol 'forward' reachable)

                              │
                              ▼
                   CLI post-compile (new):
                      if !export_functions.is_empty() { c_header::emit() }
                                          │
                                          ▼
                              libmodel.h (prototype for `forward`)
```

## 10. Risks & Open Questions

- **Risk: `ExportTypeInfo` shape extraction from type-checked signature may be incomplete for edge cases** (e.g. partial generic instantiation, nested tuples). Mitigation: semantic validation in §5 rejects unsupported shapes before codegen, so `ExportTypeInfo` only needs to handle the approved set.
- **Risk: the "first implicit `NslModel*` parameter" convention may not compose cleanly for functions that don't need model state** (pure compute kernels). Mitigation: every `@export` function takes `NslModel*` uniformly for consistency; pure-compute functions ignore the handle. C consumers can pass a dummy / null-equivalent if the runtime tolerates it, or we refine later.
- **Risk: header emission runs per compile unit; if a user compiles multiple `.nsl` files into one `.so` (future multi-module shared lib), the headers collide.** Out of scope per §2, but worth flagging.
- **Open: does the runtime's `nsl_model_forward` auto-dispatch by function name, or does the `@export` function need its own C-wrapper emission?** The existing c_api.rs has a single `nsl_model_forward` that looks up a registered forward function. `@export`-named functions like `predict` aren't routed through this. Implementation choice: either (a) emit a C-ABI wrapper per `@export` function that converts `NslTensorDesc` to internal `NslTensor` and calls the underlying function, OR (b) register each `@export` function in the runtime's function table so `nsl_model_forward` can dispatch by name. Option (a) is simpler and spec-matching; option (b) requires runtime changes out of scope. Default: **(a)** — emit per-function C wrappers at codegen time. Implementation plan's Task A2 will cover the wrapper emission.

## 11. Success Criteria

1. `nsl build --shared-lib examples/m62_shared_lib.nsl` produces a `.so` with the `add` symbol in its export table (verified via `nm` or `dumpbin`).
2. The generated `.h` contains a prototype for `add` matching its NSL signature.
3. Python can `ctypes.CDLL` the `.so`, find the `add` symbol, call it, and get a correct element-wise sum.
4. `@export(name="predict")` produces `predict` as the exported symbol instead of the NSL-side name.
5. All 15 tests in §8 pass; no existing tests regress.
6. Semantic validation catches all 9 error cases in §5.1 before codegen.
7. Non-`@export` functions in a shared-lib build remain `Linkage::Local` with mangled names (no internal-symbol leakage).

## 12. Files Touched

**Create:**
- `crates/nsl-codegen/src/c_header.rs` — header emission + `ExportTypeInfo` → C-type lowering.
- `crates/nsl-semantic/src/export.rs` — `@export` validation.
- `crates/nsl-codegen/tests/c_header_snapshot.rs` — header snapshot test.
- `crates/nsl-codegen/tests/fixtures/m62_export_header.nsl` — 2-function + 1-rename fixture.
- `examples/m62_shared_lib.nsl` — E2E fixture.
- `python/tests/test_m62_export.py` — Python E2E.

**Modify:**
- `crates/nsl-codegen/src/compiler/declaration.rs` — `@export` in decorator loop; symbol-name + linkage logic; `ExportInfo` tracking.
- `crates/nsl-codegen/src/compiler/mod.rs` — `CompilerFeatures.export_functions: Vec<ExportInfo>` field + init.
- `crates/nsl-codegen/src/lib.rs` — `pub mod c_header;`, re-export `ExportInfo`.
- `crates/nsl-semantic/src/lib.rs` — call `export::validate_exports` from top-level semantic pass.
- `crates/nsl-cli/src/main.rs` — invoke `c_header::emit` after shared-lib compile when `export_functions` non-empty.

---

**Next in M62 finish:** separate spec for fixing the grad-context bridge bugs (Bug 1: weight grads vs input grads; Bug 2: Python never enables grad; Bug 3: swallowed exception). Tracked as `2026-04-15-m62-grad-context-bridge-design.md` (not yet written).
