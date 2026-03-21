# Lightweight Effect Polymorphism — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add effect variables to function types so higher-order functions correctly propagate effects from their callable arguments. Currently `fn map(f: fn(T) -> U, xs: list[T]) -> list[U]` has no way to express that `map` inherits whatever effects `f` has. This means the effect checker either rejects valid code or misses effect violations in generic stdlib functions.

**Architecture:** A lightweight approach — not Koka's full row-polymorphic system with algebraic handlers. We add: (1) an `EffectVar` type that can appear on function types, (2) effect unification at call sites that binds effect variables to concrete effect sets, (3) effect variable propagation through the call graph. The existing bitset `EffectSet` remains the runtime representation — effect variables are resolved at compile time.

**Tech Stack:** Rust, existing `nsl-semantic/src/effects.rs`, `types.rs`, `checker/`

**Research Basis:** Type Systems & Safety notebook describes Koka's row-polymorphic effects. We implement the simplest useful subset: effect variables on function parameters with lattice-based unification (set union). No algebraic handlers, no selective CPS, no row types.

---

## Background

### The Problem

```
fn map(f: fn(int) -> int, xs: list[int]) -> list[int]:
    let result = []
    for x in xs:
        result.append(f(x))
    return result

@pure
fn double(x: int) -> int:
    return x * 2

fn noisy_double(x: int) -> int:
    print(x)          # IO effect
    return x * 2

# Today: map's effect is inferred as the union of ALL callees' effects
# This means map(double, xs) is flagged as having IO because map ALSO calls noisy_double somewhere
# In reality, map(double, xs) should be pure
```

### The Solution: Effect Variables

```
fn map<E>(f: fn(int) -> int | E, xs: list[int]) -> list[int] | E:
    ...

map(double, xs)         # E = {} → map call is pure ✓
map(noisy_double, xs)   # E = {IO} → map call has IO ✓
```

The effect variable `E` is bound at each call site to the concrete effects of the passed function.

### Current Infrastructure

- `EffectSet`: u8 bitset (IO=1, RANDOM=2, MUTATION=4, COMMUNICATION=8)
- `Type::Function { params, ret }` — no effect field
- `Type::TypeVar(Symbol)` — exists for type generics but unused
- `FnDef.type_params: Vec<TypeParam>` — exists for type generics
- Fixed-point effect propagation over call graph
- Effect checker NOT wired into semantic checker (defined but disconnected)

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/nsl-semantic/src/types.rs` | Add effect field to `Type::Function`, add `Effect` enum |
| `crates/nsl-semantic/src/effects.rs` | Add `EffectVar`, effect unification, update propagation |
| `crates/nsl-semantic/src/checker/decl.rs` | Parse effect parameters on function declarations |
| `crates/nsl-semantic/src/checker/expr.rs` | Unify effect variables at call sites |
| `crates/nsl-semantic/src/checker/mod.rs` | Wire EffectChecker into TypeChecker |
| `crates/nsl-ast/src/decl.rs` | Add `effect_params` to `FnDef` |
| `crates/nsl-parser/src/decl.rs` | Parse `<E>` effect parameter syntax |
| Tests | Effect propagation through higher-order functions |

---

## Tasks

### Task 1: Effect Type Representation

- [ ] **1.1** Add `Effect` enum to `types.rs`:
```rust
/// An effect annotation: either concrete or a variable to be unified.
#[derive(Clone, Debug, PartialEq)]
pub enum Effect {
    Concrete(EffectSet),    // known effects: {IO, RANDOM}
    Var(Symbol),            // effect variable: E (resolved at call site)
    Union(Box<Effect>, Box<Effect>),  // E | {IO} (variable plus known effects)
}

impl Effect {
    pub fn pure() -> Self { Effect::Concrete(EffectSet::empty()) }
    pub fn unknown() -> Self { Effect::Concrete(EffectSet::all()) }
}
```

- [ ] **1.2** Add effect field to `Type::Function`:
```rust
Type::Function {
    params: Vec<Type>,
    ret: Box<Type>,
    effect: Effect,          // NEW
}
```

- [ ] **1.3** Default: functions without explicit effect annotation get `Effect::unknown()` (conservative, backwards-compatible).

- [ ] **1.4** Test: construct `Type::Function` with `Effect::Var("E")`, verify equality/clone work.

### Task 2: AST Support for Effect Parameters

- [ ] **2.1** Add to `FnDef`:
```rust
pub struct FnDef {
    pub name: Symbol,
    pub type_params: Vec<TypeParam>,
    pub effect_params: Vec<Symbol>,    // NEW: effect variable names
    // ...
}
```

- [ ] **2.2** Syntax design — reuse angle brackets with `effect` keyword:
```
fn map<effect E>(f: fn(T) -> U | E, xs: list[T]) -> list[U] | E:
    ...
```

Or simpler — auto-infer effect variables from `| E` annotations without explicit declaration:
```
fn map(f: fn(T) -> U | E, xs: list[T]) -> list[U] | E:
    ...
```
Where `E` is automatically recognized as an effect variable if it appears after `|` and isn't a known effect name.

- [ ] **2.3** Parse `| E` effect suffix on function types and return types.

- [ ] **2.4** Test: parse `fn(int) -> int | E` produces `Type::Function { ..., effect: Effect::Var("E") }`.

### Task 3: Effect Unification at Call Sites

- [ ] **3.1** Add effect unification to `check_call()` in `checker/expr.rs`:
```rust
fn unify_effect(
    param_effect: &Effect,        // what the parameter expects
    arg_effect: &Effect,          // what the argument provides
    bindings: &mut HashMap<Symbol, EffectSet>,  // accumulated bindings
) -> Result<(), EffectError> {
    match (param_effect, arg_effect) {
        (Effect::Var(name), Effect::Concrete(set)) => {
            // Bind variable to concrete set
            bindings.insert(name.clone(), *set);
            Ok(())
        }
        (Effect::Concrete(expected), Effect::Concrete(actual)) => {
            // Concrete check: actual must be subset of expected
            if actual.is_subset_of(expected) { Ok(()) }
            else { Err(EffectError::Mismatch { expected, actual }) }
        }
        _ => Ok(()) // other cases: conservative pass
    }
}
```

- [ ] **3.2** After unifying all parameters, substitute bindings into the return type's effect:
```rust
// If return type has Effect::Var("E") and bindings has E → {IO}
// Then call's effect is {IO}
fn substitute_effect(effect: &Effect, bindings: &HashMap<Symbol, EffectSet>) -> EffectSet {
    match effect {
        Effect::Concrete(set) => *set,
        Effect::Var(name) => bindings.get(name).copied().unwrap_or(EffectSet::all()),
        Effect::Union(a, b) => substitute_effect(a, bindings) | substitute_effect(b, bindings),
    }
}
```

- [ ] **3.3** Test: `map(double, xs)` where `double` is `@pure` → map call effect is `{}` (pure).

- [ ] **3.4** Test: `map(noisy_double, xs)` where `noisy_double` has `{IO}` → map call effect is `{IO}`.

### Task 4: Wire Effect Checker into Semantic Checker

- [ ] **4.1** In `checker/mod.rs`, create `EffectChecker` instance alongside `TypeChecker`.

- [ ] **4.2** After type-checking each function body, register its inferred effects with the effect checker.

- [ ] **4.3** After all functions are checked, run effect propagation (existing fixed-point algorithm).

- [ ] **4.4** Validate annotations (`@pure`, `@deterministic`) against propagated effects.

- [ ] **4.5** Report effect errors as diagnostics (not panics).

### Task 5: Effect Inference for Unannotated Functions

- [ ] **5.1** Functions without explicit effect annotations: infer effects from body.
  - Direct calls to builtins: look up in `classify_builtin_effects()`
  - Calls to other user functions: union of callee's effects
  - Calls through function parameters: use the parameter's effect annotation

- [ ] **5.2** Functions with effect variables: the variable's value is determined per-call-site, not globally.

- [ ] **5.3** The existing fixed-point propagation handles non-parametric functions. For parametric functions, effects are resolved at each call site independently.

### Task 6: Stdlib Annotations

- [ ] **6.1** Annotate key stdlib functions with effect variables:
```
# stdlib/nsl/math.nsl
fn map(f: fn(T) -> U | E, xs: list[T]) -> list[U] | E:
    ...

fn filter(f: fn(T) -> bool | E, xs: list[T]) -> list[T] | E:
    ...

fn reduce(f: fn(T, T) -> T | E, xs: list[T], init: T) -> T | E:
    ...
```

- [ ] **6.2** Annotate stdlib functions with concrete effects:
```
@pure
fn relu(x: Tensor) -> Tensor:
    ...

fn dropout(x: Tensor, p: float) -> Tensor | {RANDOM}:
    ...
```

- [ ] **6.3** This is optional — unannotated functions still work (effects inferred conservatively).

### Task 7: Testing

- [ ] **7.1** Higher-order function propagation:
```
@pure
fn double(x: int) -> int:
    return x * 2

fn apply(f: fn(int) -> int | E, x: int) -> int | E:
    return f(x)

@pure
fn test_pure():
    let y = apply(double, 5)    # OK: E={} → apply is pure
```

- [ ] **7.2** Effect violation detection:
```
fn noisy(x: int) -> int:
    print(x)
    return x

@pure
fn test_violation():
    let y = apply(noisy, 5)    # ERROR: apply has IO via noisy, violates @pure
```

- [ ] **7.3** Multiple effect variables:
```
fn combine(f: fn() -> T | E1, g: fn() -> U | E2) -> (T, U) | E1 | E2:
    return (f(), g())
```

- [ ] **7.4** Backwards compatibility: existing code without effect annotations continues to work (effects inferred conservatively).

---

## Scope Limitations (Explicit Non-Goals)

- **No algebraic effect handlers**: No `handle`, `resume`, or custom control flow operators
- **No row types**: Effects are flat sets, not extensible rows
- **No selective CPS**: All code compiles to direct style (no continuation transforms)
- **No effect subtyping**: Only set membership checks, not subtype hierarchies
- **No effect inference for closures**: Closures get conservative effect (union of captured + body effects)

---

## Effort Estimate

- Task 1 (type representation): 0.5 days
- Task 2 (AST + parsing): 1 day
- Task 3 (unification at call sites): 1.5 days
- Task 4 (wire into checker): 1 day
- Task 5 (inference): 1 day
- Task 6 (stdlib annotations): 0.5 days
- Task 7 (testing): 0.5 days
- Total: **5-7 days**
