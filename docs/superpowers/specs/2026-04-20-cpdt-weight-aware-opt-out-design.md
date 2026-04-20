# CPDT `@cpdt(weight_aware=false)` Runtime Opt-Out — Design

> **Framing:** Threads the existing semantic `CpdtConfig::weight_aware` field through to the CPDT codegen pipeline so `@cpdt(weight_aware=false)` actually suppresses the weight-aware path (plan_map, validate, tier-agreement diagnostic, CPDT_CALIB_K warning). Phase 1 constraint: exactly one `@cpdt` decorator per program — enforced in the semantic pass.

**Parent specs:**
- [2026-04-18-cpdt-weight-aware-phase1-design.md](2026-04-18-cpdt-weight-aware-phase1-design.md) §1.2 item 2 (reframed monitoring-gate)
- [2026-04-19-cpdt-calibration-correction-design.md](2026-04-19-cpdt-calibration-correction-design.md)
- [2026-04-20-cpdt-validate-body-design.md](2026-04-20-cpdt-validate-body-design.md) (concurrent; PR #90)

**Worktree:** `.worktrees/cpdt-weight-aware-opt-out` (branch `feat/cpdt-weight-aware-opt-out`, based on `main`)

---

## 1. Motivation

Phase 1 ships a `weight_aware: bool` field on `CpdtConfig` (semantic crate, set via `@cpdt(weight_aware=false)`) but the parsed config is discarded — `validate_cpdt_decorator` emits diagnostics and drops the `Option<CpdtConfig>`. Result: users cannot actually suppress the weight-aware path today. The diagnostic fires regardless; the validate call (from PR #90) runs regardless; users who don't want CPDT to care about their checkpoint have no escape hatch short of disabling CPDT entirely.

The opt-out matters now because Phase 1's diagnostic is `<20%` monitoring-gate — not `<5%` hard-gate — which means on numel-degenerate architectures the diagnostic fires every build. Users who've accepted the Phase 1 information-bottleneck and don't want noise need a way to silence it for specific builds without disabling CPDT altogether.

## 2. Scope & Non-Goals

**In scope:**

- **Semantic pass:** enforce "exactly one `@cpdt` decorator per program." Multiple `@cpdt` decorators emit a Diagnostic::error naming both spans.
- **Codegen:** add `cpdt_weight_aware: bool` field to `Compiler` (default `true`). Parse the single `@cpdt` decorator's `weight_aware` kwarg during the `StmtKind::Decorated` match-arm walk; set the field accordingly.
- **Cascade skip in `invoke_cpdt_if_enabled`:** when `cpdt_weight_aware == false`, shadow `weight_map_ref` to `None` immediately after its construction. All downstream logic already guards on `weight_map_ref.is_some()` / `weights_present`; the cascade propagates without further changes (verified pre-dispatch).

**Out of scope:**

- Per-decorator-site granularity. Phase 1 treats `cpdt_weight_aware` as global-with-single-writer. If the user has two `@cpdt` decorators with different `weight_aware` settings, semantic errors before codegen runs. Phase 2 may relax this to per-decorator scope if multi-decorator programs become useful.
- Plumbing the full `CpdtConfig` struct from semantic to codegen. Only the `weight_aware` bool is needed today. The semantic pass still emits diagnostics for other CpdtConfig fields (`mode`, `cluster`, etc.) via validation-only parsing — unchanged.
- AST `load_safetensors(...)` auto-detect. Separate Tier A item (Items 3+4 in the post-retune sequence).

## 3. Design

### 3.1 Semantic pass: exactly-one-`@cpdt` enforcement

Add to `TypeChecker` (`crates/nsl-semantic/src/checker/mod.rs`):

```rust
pub struct TypeChecker<'a> {
    // ... existing fields
    /// Span of the first `@cpdt` decorator seen in this module. Phase 1
    /// requires exactly one; subsequent decorators emit a diagnostic error.
    cpdt_decorator_span: Option<nsl_ast::Span>,
}
```

Initialize to `None` in `TypeChecker::new()`.

In `crates/nsl-semantic/src/checker/stmt.rs` at the existing `@cpdt` handling site:

```rust
if dname == "cpdt" {
    let resolve = |s: nsl_ast::Symbol| -> String {
        self.interner.resolve(s.0).unwrap_or("").to_string()
    };
    // Enforce Phase 1's single-decorator constraint.
    if let Some(prev_span) = self.cpdt_decorator_span {
        self.diagnostics.push(
            Diagnostic::error(
                "@cpdt may appear at most once per program (Phase 1 restriction)"
                    .to_string(),
            )
            .with_label(deco.span, "duplicate @cpdt decorator")
            .with_label(prev_span, "previous @cpdt decorator here"),
        );
    } else {
        self.cpdt_decorator_span = Some(deco.span);
    }
    crate::cpdt::validate_cpdt_decorator(deco, &resolve, &mut self.diagnostics);
}
```

`validate_cpdt_decorator`'s return value continues to be discarded (validation diagnostics only). The Phase 1 field value flows through to codegen via the codegen's own decorator walk (§3.2), not through the semantic pass — because the semantic crate would otherwise need a cross-crate plumbing mechanism that doesn't exist for any other decorator config today.

### 3.2 Codegen: Compiler field + decorator parsing

Add to `Compiler` (`crates/nsl-codegen/src/compiler/mod.rs`):

```rust
pub struct Compiler<'a> {
    // ... existing fields
    /// Global default for CPDT's weight-aware path. Phase 1 requires exactly
    /// one `@cpdt` decorator per program (enforced in nsl-semantic); this
    /// field reflects that decorator's `weight_aware` argument, defaulting
    /// to `true`. `@cpdt(weight_aware=false)` sets it to `false`, which
    /// cascades through `invoke_cpdt_if_enabled` to suppress plan_map,
    /// validate, tier-agreement diagnostic, and CPDT_CALIB_K warning.
    ///
    /// Phase 2 may extend this to per-decorator-site settings if multi-
    /// decorator programs become supported; the single-writer semantics
    /// remain correct as a default until then.
    pub cpdt_weight_aware: bool,
}
```

Default to `true` in `Compiler::new()`.

In `StmtKind::Decorated` match arm (`crates/nsl-codegen/src/stmt.rs:1289`), add a branch for `@cpdt` alongside the existing `@no_grad` / `@fp8_compute` / `@fuse` branches:

```rust
} else if dname == "cpdt" {
    // Phase 1: read `weight_aware` kwarg from the sole @cpdt decorator
    // in the program (semantic pass has already enforced exactly-one).
    // Default (true) stays when the kwarg is absent or malformed.
    if let Some(args) = &d.args {
        for arg in args {
            if let Some(name_sym) = arg.name {
                if self.resolve_sym(name_sym) == "weight_aware" {
                    if let nsl_ast::expr::ExprKind::BoolLiteral(b) = arg.value.kind {
                        self.cpdt_weight_aware = b;
                    }
                }
            }
        }
    }
}
```

Malformed args (non-bool literal, missing value) are already diagnosed by the semantic pass; codegen silently keeps the default in that case.

### 3.3 Cascade skip

In `invoke_cpdt_if_enabled` (`crates/nsl-codegen/src/stmt.rs:75`):

```rust
// Existing:
let weight_map_ref = compiler.features.weight_map.as_ref();

// Add: Phase 1 opt-out — @cpdt(weight_aware=false) forces the weight-aware
// path off even when --weights is present. Shadowing here propagates
// through every downstream guard: plan_map receives None (returns default),
// validate's `if let Some(wm)` skips, tier-agreement diagnostic's
// `weights_present` is false, CPDT_CALIB_K warning's guard is false.
let weight_map_ref = if compiler.cpdt_weight_aware {
    weight_map_ref
} else {
    None
};

let weights_present = weight_map_ref.is_some();
```

**Cascade verification (pre-dispatch grep):**
- `plan_map` invocation in `cpdt::run:258-261` — matches on `input.weights`; `None` produces `PrecisionPlan::default()`. ✓
- Tier-agreement diagnostic in `stmt.rs:103` — guarded on `weights_present && precision_plan_built`. Shadow to `None` → `weights_present = false` → skipped. ✓
- CPDT_CALIB_K warning in `stmt.rs:134` — nested inside the tier-agreement guard. ✓
- Validate call from PR #90 (when merged) — guarded on `if let Some(wm) = weight_map_ref { ... }`. Shadow to `None` → skipped. ✓

No additional skip lines needed. The single shadow at the top of `invoke_cpdt_if_enabled` is sufficient.

## 4. Tests

Four tests:

1. **`default_is_weight_aware_true`** (unit test on Compiler): new `Compiler` instance has `cpdt_weight_aware == true`.

2. **`weight_aware_false_skips_weight_map_wiring`** (integration test): construct a Compiler with `cpdt_weight_aware = false`, `cpdt_mode = Full`, a populated `cpdt_cluster`, and a loaded `features.weight_map` (via calib_small fixture). Construct a synthetic AppliedPlan. Call `invoke_cpdt_if_enabled`. Assert:
   ```rust
   assert_eq!(compiler.cpdt_plan.as_ref().unwrap().precision.params.len(), 0,
       "weight_aware=false should produce zero weight-derived tier assignments; \
        got {} (weight-map wiring not properly skipped?)",
       compiler.cpdt_plan.as_ref().unwrap().precision.params.len());
   ```

3. **`weight_aware_true_populates_precision_plan`** (integration test): same setup but `cpdt_weight_aware = true`. Assert `params.len() >= 70` (calib_small has 74 tensors; lower bound accommodates minor fixture perturbations):
   ```rust
   assert!(plan.precision.params.len() >= 70,
       "weight-aware CPDT should produce tier assignments for ~all tensors; \
        got {} params", plan.precision.params.len());
   ```

4. **`duplicate_cpdt_decorator_emits_diagnostic`** (semantic test): parse an NSL source with two `@cpdt` decorators, run the TypeChecker, assert one of the collected diagnostics has `"@cpdt may appear at most once per program"` in its message and two labels pointing at both decorator spans.

The integration tests (2 + 3) need a full Compiler construction. Worst case they become end-to-end tests that compile a minimal NSL source and inspect the resulting `cpdt_plan`; acceptable if the helper surface is too thin for a unit-test approach.

## 5. Commit Sequencing

Single commit on `feat/cpdt-weight-aware-opt-out`:

```text
feat(cpdt): @cpdt(weight_aware=false) runtime opt-out
```

Files modified:

- `crates/nsl-semantic/src/checker/mod.rs` — `cpdt_decorator_span` field.
- `crates/nsl-semantic/src/checker/stmt.rs` — enforcement at `@cpdt` decorator site.
- `crates/nsl-codegen/src/compiler/mod.rs` — `cpdt_weight_aware` field on Compiler.
- `crates/nsl-codegen/src/stmt.rs` — `@cpdt` branch in `Decorated` arm + cascade shadow in `invoke_cpdt_if_enabled`.
- `crates/nsl-codegen/tests/cpdt_weight_aware_opt_out.rs` — new test file.
- `crates/nsl-semantic/src/cpdt.rs` — one-line doc-comment update on `validate_cpdt_decorator` noting the Phase 1 single-decorator constraint.

## 6. Phase 2 Extension Path

When multi-`@cpdt` programs become useful, the extension is:

1. Remove the semantic exactly-one enforcement (delete `cpdt_decorator_span` field + check).
2. Replace Compiler's `cpdt_weight_aware: bool` with a keyed map (e.g. by decorator-site span or by enclosing train-block name).
3. `invoke_cpdt_if_enabled` reads the per-site setting via the keying; the cascade shadow becomes per-invocation rather than global.

This is a mechanical extension. The Phase 1 design uses global-with-single-writer precisely so the extension path is clean — users who only have one decorator get Phase 2 behavior automatically.

## 7. Close-Out Criteria

- `Compiler::cpdt_weight_aware` field defaults to `true`; `@cpdt(weight_aware=false)` sets it to `false`.
- Second `@cpdt` decorator in the same program emits a Diagnostic::error with both spans labeled.
- `invoke_cpdt_if_enabled` skips `plan_map` / validate / tier-agreement diagnostic / CPDT_CALIB_K warning when `cpdt_weight_aware == false`.
- Four tests pass.
- All existing tests still pass.
- Single commit; PR opened.
