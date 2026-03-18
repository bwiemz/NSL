# M46: Reproducibility Mode — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--deterministic` compilation mode that statically detects non-deterministic operations, classifies them (auto-fixable GPU atomics, implicit RNG errors, external warnings), tracks explicit RNG seeds, and provides deterministic kernel variant selection. Compile-time guarantee — not a runtime prayer.

**Architecture:** New semantic module `determinism.rs` with `DeterminismChecker` that classifies operations and tracks RNG state. New codegen module `deterministic_kernels.rs` with kernel variant selection. New runtime stubs for deterministic reduction/scatter FFI. `@deterministic` decorator validation. `--deterministic` CLI flag. Checkpoint fingerprint computation.

**Tech Stack:** Rust (semantic analysis + codegen + runtime FFI)

**Spec:** `docs/superpowers/specs/2026-03-15-m46-reproducibility-design.md`

**Prerequisites:** None (standalone; M51 Effect System enhances but is not required)

---

## Important: Scope of This Plan

**This plan builds the determinism checking + kernel selection + RNG tracking infrastructure.** It delivers:
- `DeterminismChecker` — semantic pass classifying ops as deterministic/non-deterministic
- `DeterminismClass` enum (Deterministic, NonDeterministicGpu, NonDeterministicRng, Unknown)
- `DeterminismMode` enum (Off, FunctionLevel, Global)
- `RngState` tracking (ExplicitSeed, Derived, Implicit)
- Non-deterministic op taxonomy: GPU atomics (auto-fixable), implicit RNG (error), external (warning)
- `@deterministic` decorator semantic validation
- Deterministic kernel variant selection (`select_kernel` dispatch)
- Deterministic reduction/scatter FFI stubs
- `--deterministic` CLI flag + CompileOptions field
- Checkpoint fingerprint computation (graph hash)
- 15+ unit tests

**Deferred to M46b:** Actual deterministic GPU PTX kernels (sort-based reduction, sequential scatter), cuBLAS deterministic mode initialization, `Rng` type in type system, `train` block seed requirements, `--verify-checkpoint` subcommand, cross-function determinism propagation via call graph, E2E bit-exact reproducibility tests.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-semantic/src/determinism.rs` | `DeterminismChecker`, op classification, RNG tracking | 250 |
| `crates/nsl-codegen/src/deterministic_kernels.rs` | Kernel variant selection, graph hash computation | 150 |
| `crates/nsl-runtime/src/deterministic_ops.rs` | Deterministic reduction/scatter FFI stubs | 80 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod determinism;` |
| `crates/nsl-semantic/src/checker.rs` | Wire `@deterministic` validation |
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod deterministic_kernels;`, `deterministic` to CompileOptions |
| `crates/nsl-codegen/src/builtins.rs` | Register deterministic FFI variants |
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod deterministic_ops;` |
| `crates/nsl-cli/src/main.rs` | Add `--deterministic` flag |

---

## Phase 1: Determinism Checker

### Task 1: DeterminismChecker + Op Classification

**Files:**
- Create: `crates/nsl-semantic/src/determinism.rs`

- [ ] **Step 1: Create determinism.rs with checker, op classification, and RNG tracking**

```rust
//! M46: Compile-time determinism checking.
//!
//! Classifies operations as deterministic or non-deterministic,
//! detects implicit RNG usage, and tracks explicit seed state.

use std::collections::{HashMap, HashSet};
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// How strictly determinism is enforced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeterminismMode {
    /// No checking (default).
    Off,
    /// Only @deterministic-decorated functions are checked.
    FunctionLevel,
    /// All functions are checked (--deterministic flag).
    Global,
}

/// Classification of a function's determinism.
#[derive(Clone, Debug, PartialEq)]
pub enum DeterminismClass {
    /// No non-deterministic operations.
    Deterministic,
    /// Uses non-deterministic GPU ops (auto-fixable with deterministic kernel variants).
    NonDeterministicGpu { ops: Vec<String> },
    /// Uses implicit RNG without explicit seed (compile error).
    NonDeterministicRng { calls: Vec<String> },
    /// Contains external/unknown calls.
    Unknown,
}

/// RNG variable state tracking.
#[derive(Clone, Debug, PartialEq)]
pub enum RngState {
    /// Created with explicit seed: Rng(seed=42)
    ExplicitSeed(i64),
    /// Derived from another explicit Rng (.fork())
    Derived,
    /// Implicit/global RNG — error under --deterministic
    Implicit,
}

/// Non-deterministic operation categories.
#[derive(Clone, Debug, PartialEq)]
pub enum NonDetCategory {
    /// GPU atomic-based reduction — auto-fixable with sort-based kernel
    GpuAtomic,
    /// Implicit RNG — requires explicit seed argument
    ImplicitRng,
    /// Algorithm selection (cuBLAS heuristics) — auto-fixable
    AlgorithmSelection,
    /// External source (float non-associativity, thread scheduling) — warning only
    External,
}

/// Checks tensor operations for non-determinism.
pub struct DeterminismChecker {
    mode: DeterminismMode,
    deterministic_fns: HashSet<String>,
    rng_variables: HashMap<Symbol, RngState>,
    pub diagnostics: Vec<Diagnostic>,
}

impl DeterminismChecker {
    pub fn new(mode: DeterminismMode) -> Self {
        DeterminismChecker {
            mode,
            deterministic_fns: HashSet::new(),
            rng_variables: HashMap::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Mark a function as @deterministic.
    pub fn mark_deterministic(&mut self, name: &str) {
        self.deterministic_fns.insert(name.to_string());
    }

    /// Check if a function is marked @deterministic.
    pub fn is_deterministic_fn(&self, name: &str) -> bool {
        self.deterministic_fns.contains(name)
    }

    /// Register an RNG variable with its seed state.
    pub fn register_rng(&mut self, sym: Symbol, state: RngState) {
        self.rng_variables.insert(sym, state);
    }

    /// Classify a tensor operation by its determinism properties.
    pub fn classify_op(&self, op_name: &str) -> NonDetCategory {
        match op_name {
            // Category 1: GPU atomic reductions — auto-fixable
            "reduce_sum" | "reduce_mean" | "scatter_add" | "embedding_backward" =>
                NonDetCategory::GpuAtomic,

            // Category 2: Algorithm selection — auto-fixable
            "matmul" | "conv2d" =>
                NonDetCategory::AlgorithmSelection,

            // Category 3: Implicit RNG — compile error
            "rand" | "randn" | "dropout" | "random_normal" | "random_uniform" =>
                NonDetCategory::ImplicitRng,

            // Everything else is deterministic or external
            _ => NonDetCategory::External,
        }
    }

    /// Check if a function call is allowed under the current determinism mode.
    ///
    /// Returns a diagnostic if the call violates determinism requirements.
    pub fn check_call(
        &self,
        func_name: &str,
        has_rng_arg: bool,
        span: nsl_errors::Span,
    ) -> Option<Diagnostic> {
        if self.mode == DeterminismMode::Off {
            return None;
        }

        let category = self.classify_op(func_name);
        match category {
            NonDetCategory::ImplicitRng if !has_rng_arg => {
                Some(Diagnostic::error(format!(
                    "'{func_name}' uses implicit RNG — pass explicit 'rng: Rng' argument in deterministic mode"
                )).with_label(span, "non-deterministic call"))
            }
            NonDetCategory::GpuAtomic => {
                // Auto-fixable — emit info diagnostic
                Some(Diagnostic::warning(format!(
                    "'{func_name}' uses GPU atomics — will auto-select deterministic kernel variant"
                )).with_label(span, "auto-fixed"))
            }
            _ => None,
        }
    }

    /// Get the deterministic kernel variant name for an op.
    pub fn deterministic_variant(&self, op_name: &str) -> Option<&'static str> {
        match op_name {
            "reduce_sum" => Some("nsl_tensor_reduce_sum_deterministic"),
            "reduce_mean" => Some("nsl_tensor_reduce_mean_deterministic"),
            "scatter_add" => Some("nsl_tensor_scatter_add_deterministic"),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_gpu_atomic_ops() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.classify_op("reduce_sum"), NonDetCategory::GpuAtomic);
        assert_eq!(checker.classify_op("scatter_add"), NonDetCategory::GpuAtomic);
    }

    #[test]
    fn classify_implicit_rng_ops() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.classify_op("rand"), NonDetCategory::ImplicitRng);
        assert_eq!(checker.classify_op("dropout"), NonDetCategory::ImplicitRng);
    }

    #[test]
    fn classify_algorithm_selection_ops() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.classify_op("matmul"), NonDetCategory::AlgorithmSelection);
        assert_eq!(checker.classify_op("conv2d"), NonDetCategory::AlgorithmSelection);
    }

    #[test]
    fn classify_deterministic_ops() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.classify_op("relu"), NonDetCategory::External);
        assert_eq!(checker.classify_op("add"), NonDetCategory::External);
    }

    #[test]
    fn implicit_rng_errors_in_global_mode() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        let diag = checker.check_call("rand", false, nsl_errors::Span::dummy());
        assert!(diag.is_some());
        // With explicit rng arg → no error
        let diag = checker.check_call("rand", true, nsl_errors::Span::dummy());
        assert!(diag.is_none());
    }

    #[test]
    fn gpu_atomic_warns_in_global_mode() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        let diag = checker.check_call("reduce_sum", false, nsl_errors::Span::dummy());
        assert!(diag.is_some()); // warning, not error
    }

    #[test]
    fn no_checks_in_off_mode() {
        let checker = DeterminismChecker::new(DeterminismMode::Off);
        let diag = checker.check_call("rand", false, nsl_errors::Span::dummy());
        assert!(diag.is_none());
    }

    #[test]
    fn deterministic_variant_selection() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.deterministic_variant("reduce_sum"), Some("nsl_tensor_reduce_sum_deterministic"));
        assert_eq!(checker.deterministic_variant("relu"), None);
    }

    #[test]
    fn mark_and_query_deterministic_fn() {
        let mut checker = DeterminismChecker::new(DeterminismMode::FunctionLevel);
        assert!(!checker.is_deterministic_fn("attention"));
        checker.mark_deterministic("attention");
        assert!(checker.is_deterministic_fn("attention"));
    }

    #[test]
    fn rng_state_tracking() {
        let mut checker = DeterminismChecker::new(DeterminismMode::Global);
        let sym = nsl_ast::Symbol(nsl_lexer::Interner::new().get_or_intern("rng"));
        checker.register_rng(sym, RngState::ExplicitSeed(42));
        assert_eq!(checker.rng_variables.get(&sym), Some(&RngState::ExplicitSeed(42)));
    }
}
```

### Task 2: Deterministic Kernel Selection + Graph Hash

**Files:**
- Create: `crates/nsl-codegen/src/deterministic_kernels.rs`

- [ ] **Step 2: Create deterministic_kernels.rs with kernel dispatch and graph hash**

```rust
//! M46: Deterministic kernel variant selection and graph fingerprinting.

/// Kernel variant for deterministic mode.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelVariant {
    Default,
    DeterministicSortReduce,
    DeterministicSortAccumulate,
    DeterministicCublas,
}

/// Select the appropriate kernel variant based on determinism mode.
pub fn select_kernel(op_name: &str, deterministic: bool) -> KernelVariant {
    if !deterministic {
        return KernelVariant::Default;
    }
    match op_name {
        "reduce_sum" | "reduce_mean" => KernelVariant::DeterministicSortReduce,
        "scatter_add" | "embedding_backward" => KernelVariant::DeterministicSortAccumulate,
        "matmul" | "conv2d" => KernelVariant::DeterministicCublas,
        _ => KernelVariant::Default,
    }
}

/// Compute a deterministic hash of a computation graph for checkpoint fingerprinting.
///
/// Uses a simple string-based hash of operation names and shapes.
/// Full structural hashing (BLAKE3) deferred to M46b.
pub fn compute_graph_hash(op_sequence: &[(&str, &[usize])]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (op_name, shape) in op_sequence {
        op_name.hash(&mut hasher);
        shape.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_variant_when_not_deterministic() {
        assert_eq!(select_kernel("reduce_sum", false), KernelVariant::Default);
        assert_eq!(select_kernel("matmul", false), KernelVariant::Default);
    }

    #[test]
    fn deterministic_variant_for_atomic_ops() {
        assert_eq!(select_kernel("reduce_sum", true), KernelVariant::DeterministicSortReduce);
        assert_eq!(select_kernel("scatter_add", true), KernelVariant::DeterministicSortAccumulate);
    }

    #[test]
    fn deterministic_variant_for_cublas() {
        assert_eq!(select_kernel("matmul", true), KernelVariant::DeterministicCublas);
    }

    #[test]
    fn default_variant_for_deterministic_ops() {
        assert_eq!(select_kernel("relu", true), KernelVariant::Default);
        assert_eq!(select_kernel("add", true), KernelVariant::Default);
    }

    #[test]
    fn graph_hash_deterministic() {
        let ops1 = vec![("matmul", &[32, 128][..]), ("relu", &[32, 128][..])];
        let ops2 = vec![("matmul", &[32, 128][..]), ("relu", &[32, 128][..])];
        assert_eq!(compute_graph_hash(&ops1), compute_graph_hash(&ops2));
    }

    #[test]
    fn graph_hash_changes_with_ops() {
        let ops1 = vec![("matmul", &[32, 128][..])];
        let ops2 = vec![("relu", &[32, 128][..])];
        assert_ne!(compute_graph_hash(&ops1), compute_graph_hash(&ops2));
    }
}
```

---

## Phase 2: Runtime + Semantic + CLI

### Task 3: Runtime Deterministic FFI Stubs

**Files:**
- Create: `crates/nsl-runtime/src/deterministic_ops.rs`

- [ ] **Step 3: Create deterministic_ops.rs with FFI stubs**

```rust
//! M46: Deterministic runtime operation variants (stubs).
//!
//! These are called when --deterministic is active, replacing the default
//! non-deterministic GPU kernels. Full implementations (sort-based reduction
//! PTX) deferred to M46b.

/// Deterministic reduce_sum — uses sort-based reduction instead of atomicAdd.
/// Stub: delegates to existing nsl_tensor_sum_dim (CPU path is already deterministic).
/// Full sort-based GPU PTX in M46b.
///
/// NOTE: Signature matches nsl_tensor_sum_dim(tensor_ptr, dim, keepdim) = 3 params.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_sum_deterministic(input: i64, dim: i64, keepdim: i64) -> i64 {
    // Delegate to standard path — CPU reductions are already deterministic.
    // GPU sort-based reduction PTX deferred to M46b.
    crate::tensor::nsl_tensor_sum_dim(input, dim, keepdim)
}

/// Deterministic reduce_mean — sort-based reduction.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_mean_deterministic(input: i64, dim: i64, keepdim: i64) -> i64 {
    crate::tensor::nsl_tensor_mean_dim(input, dim, keepdim)
}

/// Deterministic scatter_add — sort indices then sequential accumulate.
/// SAFETY: Stub — returns 0 (null). Must not be called until M46b implements.
#[no_mangle]
pub extern "C" fn nsl_tensor_scatter_add_deterministic(
    _input: i64, _indices: i64, _src: i64,
) -> i64 {
    // TODO M46b: implement sort-indices-then-sequential-accumulate
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_scatter_stub_returns_zero() {
        // Stub — just verify it doesn't panic
        assert_eq!(nsl_tensor_scatter_add_deterministic(0, 0, 0), 0);
    }
}
```

**Note:** The actual reduction functions are `crate::tensor::nsl_tensor_sum_dim(ptr, dim, keepdim)` and `crate::tensor::nsl_tensor_mean_dim(ptr, dim, keepdim)` — 3 params each. The stubs delegate to these. `scatter_add` has no existing equivalent, so it returns 0 (null) with a safety comment.

### Task 4: Wire Modules + CLI + Builtins

- [ ] **Step 4: Wire all modules, add CLI flag, register builtins**

Semantic: `pub mod determinism;` in lib.rs, wire `@deterministic` in checker.rs.
Codegen: `pub mod deterministic_kernels;` in lib.rs, add `deterministic: bool` to CompileOptions.
Runtime: `pub mod deterministic_ops;` in lib.rs.
CLI: `--deterministic` flag on Run, Build, Check commands.
Builtins: register 3 deterministic FFI variants (reduce_sum/mean take 3 i64 params: ptr, dim, keepdim; scatter_add takes 3: input, indices, src).

---

## Phase 3: Build Verification

- [ ] **Step 5: `cargo build`**
- [ ] **Step 6: `cargo test` — expect 15+ new tests**
- [ ] **Step 7: `cargo clippy`**

---

## Verification Checklist

1. **Op classification**: GPU atomics → GpuAtomic, implicit RNG → ImplicitRng, matmul → AlgorithmSelection
2. **Implicit RNG error**: `rand()` without rng arg → error in Global mode
3. **GPU atomic warning**: `reduce_sum` → warning (auto-fixable) in Global mode
4. **Off mode**: No diagnostics when mode is Off
5. **Deterministic variant selection**: reduce_sum → DeterministicSortReduce, matmul → DeterministicCublas
6. **Graph hash**: Same ops → same hash, different ops → different hash
7. **@deterministic decorator**: Functions can be marked and queried
8. **RNG state tracking**: ExplicitSeed registered and queryable
9. **No regressions**: All 651+ existing tests pass
