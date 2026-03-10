# M13: Import System / Module System — Design Document

## Goal

Add a file-based module system to NSL so that code can be organized across multiple files and a standard library can be written in NSL itself. This unblocks all future milestones that need stdlib modules (optimizers, loss functions, nn layers, etc.).

## Scope

**In scope:**
- `from nsl.math import clamp, lerp` — import specific symbols from a module
- `import nsl.math as math` — import module under an alias (`math.clamp(...)`)
- Relative imports: `from ./helpers import my_func`
- Module resolution: `NSL_STDLIB_PATH` env var → `<exe_dir>/stdlib/` fallback
- Circular import detection (compile error)
- Duplicate import deduplication
- Name mangling for Cranelift symbol uniqueness
- Initial stdlib: `stdlib/nsl/math.nsl` with basic math helpers
- All top-level `fn` and `model` definitions are importable (no pub/private)

**Out of scope (future milestones):**
- `import nsl.math` without alias (requires module objects at runtime)
- `__init__.nsl` package files
- `pub`/private visibility modifiers
- Wildcard imports (`from nsl.math import *`)

## Architecture

### Module Resolution

Resolution order for `from nsl.math import clamp`:

1. Check `NSL_STDLIB_PATH` environment variable (if set) → `$NSL_STDLIB_PATH/nsl/math.nsl`
2. Fall back to `<exe_dir>/stdlib/` → `<exe_dir>/stdlib/nsl/math.nsl`

Resolution for relative imports (`from ./helpers import my_func`):

3. Resolve relative to the importing file's directory → `<dir>/helpers.nsl`

Dotted paths map to directory separators: `nsl.math` → `nsl/math.nsl`.

### Name Mangling

Cranelift uses a flat global symbol namespace. To avoid collisions between same-named functions in different modules:

- `nsl/math.nsl` → `fn clamp` → Cranelift symbol `nsl_math__clamp`
- `nsl/util.nsl` → `fn clamp` → Cranelift symbol `nsl_util__clamp`
- Main file functions remain unmangled (e.g., `fn main_helper` → `main_helper`)

Mangling scheme: replace `/` and `.` with `_`, join with `__` separator.
Example: module path `nsl/math` + function `clamp` → `nsl_math__clamp`.

The compiler maintains a mapping from local import names to mangled Cranelift symbols.

### Compilation Pipeline

The current single-file pipeline becomes multi-file with strict pass separation:

```
Pass 1 — Parse & Resolve
  1a. Parse main file
  1b. Collect all import/from-import statements
  1c. Recursively resolve and parse all imported modules
  1d. Build dependency DAG, topological sort
  1e. Detect circular imports → compile error

Pass 2 — Semantic Analysis
  For each module in topological order:
  2a. Type-check the module's AST
  2b. Build ModuleInterface (exported function signatures, model definitions)
  2c. When checking a module that imports from another, use the already-built
      ModuleInterface to resolve imported symbol types
  Fail fast on any type error across any module.

Pass 3 — Codegen (only if Pass 2 succeeds completely)
  3a. Declare all mangled function signatures across all modules in Cranelift
  3b. Compile all function bodies across all modules
  3c. Compile main file's top-level statements (main function)
  3d. Link into single binary
```

### Data Structures

```
ModuleInterface {
    path: PathBuf,                              // e.g., stdlib/nsl/math.nsl
    module_path: String,                        // e.g., "nsl.math"
    exports: HashMap<String, ExportedSymbol>,   // function name → signature info
    ast: Vec<Stmt>,                             // retained for codegen pass
}

ExportedSymbol {
    kind: SymbolKind,           // Function, Model
    mangled_name: String,       // e.g., "nsl_math__clamp"
    signature: FnSignature,     // param types + return type (for type checking)
}
```

The compiler gains:
- `modules: HashMap<PathBuf, ModuleInterface>` — all compiled modules
- `module_aliases: HashMap<Symbol, String>` — alias → module_path (for `import ... as`)
- `import_map: HashMap<Symbol, String>` — local name → mangled name (for `from ... import`)

### Import Syntax Handling

**`from nsl.math import clamp, lerp`:**
- Resolve `nsl.math` → parse `stdlib/nsl/math.nsl`
- Find `fn clamp` and `fn lerp` in that module's exports
- Inject into caller's semantic scope: `clamp` → type from ModuleInterface
- In codegen: calls to `clamp(...)` emit calls to `nsl_math__clamp`

**`import nsl.math as math`:**
- Same resolution and compilation as above
- Register alias: `math` → module "nsl.math"
- In codegen: `math.clamp(...)` → look up `clamp` in "nsl.math" exports → emit `nsl_math__clamp`

### Duplicate Import Handling

A `compiled_modules: HashSet<PathBuf>` (canonicalized paths) tracks what's been processed. If `a.nsl` and `b.nsl` both import from `nsl.math`, the module is parsed and compiled only once.

### Error Handling

- **Module not found:** `error: module 'nsl.foo' not found (searched: <paths>)`
- **Symbol not found:** `error: 'nsl.math' has no export named 'nonexistent'`
- **Circular import:** `error: circular import detected: a.nsl → b.nsl → a.nsl`

### Initial Standard Library

`stdlib/nsl/math.nsl`:
```nsl
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

### Test Examples

**Test 1: Basic from-import**
```nsl
from nsl.math import clamp, lerp
print(clamp(5.0, 0.0, 3.0))   # 3.0
print(lerp(0.0, 10.0, 0.5))   # 5.0
```

**Test 2: Alias import**
```nsl
import nsl.math as math
print(math.sign(-42.0))   # -1.0
```

**Test 3: Relative import**
```nsl
# helpers.nsl (in same directory):
# fn double(x: float) -> float:
#     return x * 2.0

from ./helpers import double
print(double(21.0))   # 42.0
```

**Test 4: Imported function used in grad block**
```nsl
from nsl.math import clamp
let w = ones([3])
let (loss, grads) = grad(w):
    let y = w * ones([3])
    y.sum()
print(clamp(loss.item(), 0.0, 100.0))  # 3.0
```

## Roadmap Impact

The roadmap shifts by one milestone:
- **M13**: Import system (this design)
- **M14**: Train block DSL + optimizers/schedulers/loss functions in NSL
- **M15+**: Everything else shifts by one number

## Design Tensions

1. **Single binary vs separate compilation units**: We compile everything into one Cranelift ObjectModule. This is simpler than producing separate .o files and linking, but means recompilation of all modules on every build. Acceptable for M13; incremental compilation can come later.

2. **No privacy model**: All top-level symbols are importable. This is intentional for M13 simplicity. A `pub` keyword can be added in a future milestone.

3. **No package-level imports**: `import nsl.math` (without `as`) would require module objects at runtime. Deferred — `import ... as` and `from ... import` cover the practical use cases.
