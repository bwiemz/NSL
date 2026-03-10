# M10: Compile-Time Tensor Type System — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the compiler infer and propagate concrete tensor shapes through operations, catching shape mismatches at compile time.

**Architecture:** The parser, type resolver, and shape algebra already exist from M9. The core gap is that tensor creation functions (`zeros`, `ones`, etc.) return `Shape::unknown()`, so shape information never enters the pipeline. M10 adds shape inference at creation sites, propagates shapes through operations (arithmetic, matmul, reshape, transpose), and produces clear diagnostics on mismatch. No runtime changes needed — this is purely compiler work.

**Tech Stack:** Rust (nsl-semantic crate primarily, minor nsl-codegen updates)

**Key Insight:** ~80% of the infrastructure already exists. The main work is:
1. Extracting shapes from literal arguments at tensor creation call sites
2. Computing result shapes for reshape/transpose methods
3. Adding compile-error test programs
4. Improving diagnostic messages

---

## Current State (What Already Works)

| Component | File | Status |
|-----------|------|--------|
| `Tensor<[dims], dtype, device>` parsing | `nsl-parser/src/types.rs:127-156` | Complete |
| Type resolver (TypeExpr → Type::Tensor) | `nsl-semantic/src/resolve.rs:18-62` | Complete |
| Shape algebra (elementwise, matmul, unify_dim) | `nsl-semantic/src/shapes.rs` | Complete |
| Tensor binary op type checking | `nsl-semantic/src/checker.rs:834-932` | Complete (but shapes are always unknown) |
| Tensor method type checking | `nsl-semantic/src/checker.rs:1058-1087` | Partial (methods return correct Function types) |
| Type annotations on variables | `nsl-semantic/src/checker.rs:297-330` | Complete (is_assignable validates) |
| Tensor creation builtins | `nsl-semantic/src/builtins.rs:71-101` | Returns `Shape::unknown()` — **this is the gap** |

## What M10 Adds

1. **Shape inference**: `zeros([3, 4])` → `Tensor<[3, 4], f64, cpu>` (not unknown)
2. **Shape propagation**: `zeros([3, 4]) + ones([3, 4])` → `Tensor<[3, 4], f64, cpu>`
3. **Shape errors**: `zeros([3, 4]) @ ones([5, 2])` → compile error (inner dims 4 ≠ 5)
4. **Reshape tracking**: `t.reshape([2, 6])` → `Tensor<[2, 6], f64, cpu>`
5. **Transpose tracking**: `t.transpose(0, 1)` on `[3, 4]` → `Tensor<[4, 3], f64, cpu>`
6. **Type annotations**: `let x: Tensor<[3, 4], f32, cpu> = zeros([3, 4])` validated
7. **Symbolic dim tracking**: `fn f(a: Tensor<[batch, 64], f64>) → batch` tracked consistently
8. **Better diagnostics**: Shape errors show actual vs expected shapes clearly

## What M10 Does NOT Add (Deferred)

- `.to(dtype)` / `.to(device)` casting methods (M11+)
- Named dimension operations like `.sum(dim="heads")` (M11+)
- `@broadcast` annotation (M11+)
- `broadcast_to()`, `unsqueeze()`, `cat()`, `squeeze()` methods (M11+)
- Sparse tensor runtime (M16)
- Element count validation for reshape (needs compile-time arithmetic on symbolics)

---

## Task 1: Shape Inference from Tensor Creation Calls

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs` (in `check_call`, ~line 947)
- Test: `crates/nsl-semantic/src/shapes.rs` (add unit tests)

This is the keystone task — everything else depends on shapes being known.

**Step 1: Write failing test program**

Create `examples/m10_shape_check.nsl`:
```nsl
# M10 Shape Checking Demo

# Shape inference from creation
let a = zeros([3, 4])
let b = ones([3, 4])

# Shape propagation through arithmetic
let c = a + b
print(c)

# Matmul with known shapes
let x = ones([3, 4])
let y = ones([4, 2])
let z = x @ y
print(z)

# Scalar-tensor ops preserve shape
let d = a * 2.0
print(d)
```

**Step 2: Verify it compiles and runs (baseline — shapes are unknown, no errors)**

Run: `cargo run -p nsl-cli -- run examples/m10_shape_check.nsl`
Expected: Compiles and runs (shapes are unknown, checking is skipped)

**Step 3: Add `extract_shape_from_list_literal` helper to checker**

In `crates/nsl-semantic/src/checker.rs`, add a helper method that inspects a call's arguments to extract a concrete shape from a list literal:

```rust
/// Try to extract a concrete tensor shape from a list-literal argument.
/// Returns Shape::unknown() if the argument isn't a list literal or contains non-integer elements.
fn extract_shape_from_args(&self, args: &[Arg]) -> Shape {
    if args.is_empty() {
        return Shape::unknown();
    }
    // First argument should be the shape list
    match &args[0].value.kind {
        ExprKind::ListLiteral(elems) => {
            let mut dims = Vec::new();
            for elem in elems {
                match &elem.kind {
                    ExprKind::IntLiteral(n) => dims.push(Dim::Concrete(*n)),
                    ExprKind::Ident(sym) => dims.push(Dim::Symbolic(*sym)),
                    _ => return Shape::unknown(), // complex expression, can't infer
                }
            }
            Shape { dims }
        }
        _ => Shape::unknown(),
    }
}
```

**Step 4: Add tensor creation shape inference in `check_call`**

In `check_call` (around line 954, after the `enumerate`/`zip` special cases), add shape inference for tensor creation functions:

```rust
// Tensor creation shape inference
if matches!(name.as_str(), "zeros" | "ones" | "rand" | "randn" | "empty") {
    let shape = self.extract_shape_from_args(args);
    return Type::Tensor {
        shape,
        dtype: DType::F64,
        device: Device::Cpu,
    };
}
if name == "full" {
    let shape = self.extract_shape_from_args(args);
    return Type::Tensor {
        shape,
        dtype: DType::F64,
        device: Device::Cpu,
    };
}
if name == "arange" {
    // arange produces 1D tensor, size unknown at compile time
    return Type::Tensor {
        shape: Shape::unknown(),
        dtype: DType::F64,
        device: Device::Cpu,
    };
}
```

**Step 5: Verify existing tests still pass**

Run: `cargo test -p nsl-semantic`
Expected: All existing tests pass

**Step 6: Verify shape inference works by adding a shape-mismatch test**

Create `examples/m10_shape_errors.nsl`:
```nsl
# This file intentionally has shape errors — used for testing diagnostics

# Elementwise rank mismatch
let a = zeros([3, 4])
let b = zeros([3, 4, 5])
let c = a + b

# Matmul inner dim mismatch
let x = zeros([3, 4])
let y = zeros([5, 2])
let z = x @ y
```

Run: `cargo run -p nsl-cli -- check examples/m10_shape_errors.nsl`
Expected: Two shape errors reported (rank mismatch + matmul inner dim mismatch)

**Step 7: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs examples/m10_shape_check.nsl examples/m10_shape_errors.nsl
git commit -m "feat(m10): infer tensor shapes from creation call arguments"
```

---

## Task 2: Tensor Type Annotations

**Files:**
- Modify: `crates/nsl-semantic/src/types.rs` (improve `is_assignable` for tensor types)
- Test: `examples/m10_shape_check.nsl` (extend)

Type annotations like `let x: Tensor<[3, 4], f32, cpu> = zeros([3, 4])` already parse and resolve. This task verifies it works end-to-end and fixes any gaps in `is_assignable`.

**Step 1: Check current `is_assignable` for Tensor types**

Read `crates/nsl-semantic/src/types.rs`, the `is_assignable` function. Verify it handles:
- `Tensor<[3, 4], f64, cpu>` assignable to `Tensor<[3, 4], f64, cpu>` (exact match)
- `Tensor<unknown, f64, cpu>` assignable to `Tensor<[3, 4], f64, cpu>` (unknown ≈ wildcard)
- `Tensor<[3, 4], f64, cpu>` NOT assignable to `Tensor<[3, 5], f64, cpu>` (dim mismatch)
- Dtype widening rules (f32 tensor assignable to f64 annotation?)

**Step 2: Add tensor assignability logic if missing**

In `is_assignable`, add a case for tensor-to-tensor assignability that uses `unify_dim` for shape checking:

```rust
// Tensor assignability: shapes must be compatible, dtype must be assignable
(
    Type::Tensor { shape: vs, dtype: vd, device: vdev },
    Type::Tensor { shape: as_, dtype: ad, device: adev },
) => {
    // Unknown shape is always compatible
    if vs.rank() == 0 || as_.rank() == 0 {
        return true;
    }
    // Ranks must match
    if vs.rank() != as_.rank() {
        return false;
    }
    // Each dimension must unify
    for (v, a) in vs.dims.iter().zip(as_.dims.iter()) {
        if shapes::unify_dim(v, a).is_none() {
            return false;
        }
    }
    // Device: unknown matches anything
    if !matches!(vdev, Device::Unknown) && !matches!(adev, Device::Unknown) && vdev != adev {
        return false;
    }
    true
}
```

**Step 3: Add annotated tensor declarations to test program**

Extend `examples/m10_shape_check.nsl`:
```nsl
# Type annotations
let w: Tensor<[4, 2], f64> = ones([4, 2])
print(w)
```

**Step 4: Add annotation mismatch to error test**

Extend `examples/m10_shape_errors.nsl`:
```nsl
# Type annotation mismatch
let bad: Tensor<[3, 4], f64> = zeros([5, 6])
```

Run: `cargo run -p nsl-cli -- check examples/m10_shape_errors.nsl`
Expected: Three errors total (rank mismatch, matmul mismatch, annotation mismatch)

**Step 5: Commit**

```bash
git add crates/nsl-semantic/src/types.rs examples/m10_shape_check.nsl examples/m10_shape_errors.nsl
git commit -m "feat(m10): tensor type annotation validation with shape checking"
```

---

## Task 3: Reshape Shape Tracking

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs` (in `check_member_access`, tensor methods section)

**Step 1: Update reshape return type to extract shape from argument**

The current code returns `obj_ty.clone()` for reshape. We need to intercept the reshape call in `check_call` after member access resolution, OR compute the shape in `check_member_access`.

The cleanest approach: in `check_call`, when the callee is a `MemberAccess` with method name `reshape` on a tensor, extract the shape from the list literal argument.

Add after the tensor creation inference block in `check_call`:

```rust
// Tensor method shape inference
if let ExprKind::MemberAccess { object, member } = &callee.kind {
    let obj_ty = self.type_map.get(&object.id).cloned().unwrap_or(Type::Unknown);
    if obj_ty.is_tensor() {
        let method_name = self.interner.resolve(member.0).unwrap_or("").to_string();
        if let Type::Tensor { dtype, device, .. } = &obj_ty {
            match method_name.as_str() {
                "reshape" => {
                    let shape = self.extract_shape_from_args(args);
                    return Type::Tensor {
                        shape,
                        dtype: *dtype,
                        device: device.clone(),
                    };
                }
                "transpose" => {
                    // Handle in Task 4
                }
                _ => {}
            }
        }
    }
}
```

**Step 2: Add reshape test to m10_shape_check.nsl**

```nsl
# Reshape with known shape
let r = z.reshape([6, 1])
print(r)
```

**Step 3: Run and verify**

Run: `cargo run -p nsl-cli -- run examples/m10_shape_check.nsl`
Expected: Compiles and runs. Internally, `r` should have type `Tensor<[6, 1], f64, cpu>`.

**Step 4: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs examples/m10_shape_check.nsl
git commit -m "feat(m10): track tensor shape through reshape calls"
```

---

## Task 4: Transpose Shape Tracking

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs` (extend the tensor method inference from Task 3)

**Step 1: Add transpose shape computation**

In the `check_call` tensor method inference block (from Task 3), fill in the transpose case:

```rust
"transpose" => {
    if let Type::Tensor { shape, dtype, device } = &obj_ty {
        if shape.rank() >= 2 && args.len() >= 2 {
            // Extract dimension indices from literal int args
            let d0 = match &args[0].value.kind {
                ExprKind::IntLiteral(n) => Some(*n as usize),
                _ => None,
            };
            let d1 = match &args[1].value.kind {
                ExprKind::IntLiteral(n) => Some(*n as usize),
                _ => None,
            };
            if let (Some(d0), Some(d1)) = (d0, d1) {
                if d0 < shape.rank() && d1 < shape.rank() {
                    let mut new_dims = shape.dims.clone();
                    new_dims.swap(d0, d1);
                    return Type::Tensor {
                        shape: Shape { dims: new_dims },
                        dtype: *dtype,
                        device: device.clone(),
                    };
                }
            }
        }
        // Can't determine statically — return unknown shape
        return Type::Tensor {
            shape: Shape::unknown(),
            dtype: *dtype,
            device: device.clone(),
        };
    }
}
```

**Step 2: Add transpose test**

Extend `examples/m10_shape_check.nsl`:
```nsl
# Transpose with known shape
let t = x.transpose(0, 1)
# t should be Tensor<[4, 3]>
let check = t @ ones([3, 2])
print(check)
```

**Step 3: Add transpose shape error test**

Extend `examples/m10_shape_errors.nsl`:
```nsl
# Matmul after transpose — inner dims should mismatch
let p = zeros([3, 4])
let q = zeros([5, 6])
let r = p.transpose(0, 1) @ q
# p.transpose(0,1) = [4, 3], q = [5, 6] → inner 3 ≠ 5 → error
```

Run: `cargo run -p nsl-cli -- check examples/m10_shape_errors.nsl`
Expected: Four errors total

**Step 4: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs examples/m10_shape_check.nsl examples/m10_shape_errors.nsl
git commit -m "feat(m10): track tensor shape through transpose calls"
```

---

## Task 5: Symbolic Dimension Consistency

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs` (function parameter scope)
- Test: `examples/m10_symbolic_dims.nsl` (new)

Symbolic dimensions (e.g., `batch`) in function signatures should be tracked so the compiler knows two parameters sharing a symbolic dim have the same size.

**Step 1: Create test program with symbolic dims**

Create `examples/m10_symbolic_dims.nsl`:
```nsl
# Symbolic dimensions in function signatures
fn matmul_check(a: Tensor<[3, 4], f64>, b: Tensor<[4, 2], f64>) -> Tensor<[3, 2], f64>:
    return a @ b

let result = matmul_check(ones([3, 4]), ones([4, 2]))
print(result)
```

**Step 2: Verify it works**

Run: `cargo run -p nsl-cli -- run examples/m10_symbolic_dims.nsl`
Expected: Compiles and runs, printing tensor([[8.0, 8.0], ...])

Note: Full symbolic dimension unification (where `batch` in param A must equal `batch` in param B) requires tracking a symbol→Dim mapping during function body checking. This is a stretch goal. The minimum for M10 is that concrete-annotated shapes in function signatures flow through operations correctly.

**Step 3: Add function with mismatched shapes to error test**

Create `examples/m10_fn_shape_errors.nsl`:
```nsl
# Function with shape-annotated parameters
fn bad_add(a: Tensor<[3, 4], f64>, b: Tensor<[5, 6], f64>) -> Tensor<[3, 4], f64>:
    return a + b

let r = bad_add(zeros([3, 4]), zeros([5, 6]))
```

Run: `cargo run -p nsl-cli -- check examples/m10_fn_shape_errors.nsl`
Expected: Shape mismatch error inside function body (elementwise [3,4] vs [5,6])

**Step 4: Commit**

```bash
git add examples/m10_symbolic_dims.nsl examples/m10_fn_shape_errors.nsl
git commit -m "feat(m10): verify shape checking in function signatures"
```

---

## Task 6: Improved Shape Error Diagnostics

**Files:**
- Modify: `crates/nsl-semantic/src/shapes.rs` (improve error messages)
- Modify: `crates/nsl-semantic/src/checker.rs` (improve tensor error context)

**Step 1: Improve `fmt_dim` to show symbolic names**

In `crates/nsl-semantic/src/shapes.rs`, update `fmt_dim`:

```rust
fn fmt_dim(d: &Dim) -> String {
    match d {
        Dim::Concrete(n) => n.to_string(),
        Dim::Symbolic(_) => "<sym>".into(), // TODO: resolve name from interner in M11+
        Dim::Named { .. } => "<named>".into(),
        Dim::Wildcard => "_".into(),
    }
}
```

**Step 2: Add `fmt_shape` helper for full shape display**

```rust
pub fn fmt_shape(s: &Shape) -> String {
    if s.rank() == 0 {
        return "[?]".into();
    }
    let dims: Vec<String> = s.dims.iter().map(|d| fmt_dim(d)).collect();
    format!("[{}]", dims.join(", "))
}
```

**Step 3: Update error messages to use `fmt_shape`**

In `check_elementwise`:
```rust
return Err(Diagnostic::error(format!(
    "shape mismatch: left has shape {}, right has shape {}",
    fmt_shape(lhs),
    fmt_shape(rhs)
))
.with_label(op_span, "incompatible shapes for element-wise operation"));
```

In `check_matmul`:
```rust
return Err(Diagnostic::error(format!(
    "matmul inner dimensions don't match: left is {}, right is {}",
    fmt_shape(lhs),
    fmt_shape(rhs)
))
.with_label(op_span, format!(
    "inner dims: {} vs {}",
    fmt_dim(l_inner),
    fmt_dim(r_inner)
)));
```

**Step 4: Verify diagnostics output**

Run: `cargo run -p nsl-cli -- check examples/m10_shape_errors.nsl`
Expected: Error messages now show actual shapes like `[3, 4]` instead of just rank numbers.

**Step 5: Commit**

```bash
git add crates/nsl-semantic/src/shapes.rs crates/nsl-semantic/src/checker.rs
git commit -m "feat(m10): improved shape error diagnostics with visual shape display"
```

---

## Task 7: Regression Tests and Baselines

**Files:**
- Create: `tests/expected/m10_shape_check.txt`
- Verify: All existing examples still compile and run

**Step 1: Run all existing examples as regression tests**

Run each example and compare with expected output:
```bash
cargo run -p nsl-cli -- run examples/hello.nsl
cargo run -p nsl-cli -- run examples/features.nsl
cargo run -p nsl-cli -- run examples/m5_features.nsl
cargo run -p nsl-cli -- run examples/m6_features.nsl
cargo run -p nsl-cli -- run examples/m8_features.nsl
cargo run -p nsl-cli -- run examples/m9_tensors.nsl
```

Expected: All produce identical output to `tests/expected/*.txt` baselines (use `diff --strip-trailing-cr`)

**Step 2: Create baseline for m10_shape_check.nsl**

Run: `cargo run -p nsl-cli -- run examples/m10_shape_check.nsl > tests/expected/m10_shape_check.txt`

**Step 3: Run `cargo test` across all crates**

Run: `cargo test --workspace`
Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/expected/m10_shape_check.txt
git commit -m "test(m10): add baselines and verify all regressions pass"
```

---

## Task 8: `--dump-types` Flag Enhancement

**Files:**
- Modify: `crates/nsl-cli/src/main.rs` (enhance `--dump-types` output)
- Modify: `crates/nsl-semantic/src/types.rs` (add Display impl or formatting)

**Step 1: Add human-readable tensor type formatting**

Add a `display_type` function (or `Display` impl) to `types.rs` that formats tensors nicely:
```
Tensor<[3, 4], f64, cpu>
```
instead of the current Debug output:
```
Tensor { shape: Shape { dims: [Concrete(3), Concrete(4)] }, dtype: F64, device: Cpu }
```

**Step 2: Update `--dump-types` in `frontend()` to use the new formatting**

Check how `--dump-types` currently works in `main.rs` and update it to show the human-readable format.

**Step 3: Verify**

Run: `cargo run -p nsl-cli -- check --dump-types examples/m10_shape_check.nsl`
Expected: Type map shows `Tensor<[3, 4], f64, cpu>` for relevant expressions

**Step 4: Commit**

```bash
git add crates/nsl-semantic/src/types.rs crates/nsl-cli/src/main.rs
git commit -m "feat(m10): human-readable tensor type display for --dump-types"
```

---

## Task 9: Update Memory and Documentation

**Files:**
- Modify: `~/.claude/projects/c--Users-bwiem-projects-NSL/memory/nsl-architecture.md`
- Modify: `~/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md`

**Step 1: Update architecture memory with M10 details**

Add M10 section to `nsl-architecture.md` documenting:
- Shape inference from tensor creation calls
- Shape propagation through binary ops, reshape, transpose
- Tensor type annotation validation
- `fmt_shape` diagnostic helper
- Key files modified

**Step 2: Update MEMORY.md roadmap**

Mark M10 as COMPLETE in the roadmap section.

**Step 3: Commit memory updates**

No git commit needed for memory files (they're outside the repo).

---

## Verification Checklist

Before M10 is considered complete:

- [ ] `zeros([3, 4])` infers `Tensor<[3, 4], f64, cpu>` (not unknown)
- [ ] `zeros([3, 4]) + ones([3, 4])` produces `Tensor<[3, 4], f64, cpu>`
- [ ] `zeros([3, 4]) @ ones([4, 2])` produces `Tensor<[3, 2], f64, cpu>`
- [ ] `zeros([3, 4]) @ ones([5, 2])` produces compile error (inner dims 4 ≠ 5)
- [ ] `zeros([3, 4]) + ones([3, 4, 5])` produces compile error (rank mismatch)
- [ ] `.reshape([2, 6])` produces `Tensor<[2, 6], ...]`
- [ ] `.transpose(0, 1)` on `[3, 4]` produces `Tensor<[4, 3], ...]`
- [ ] `let x: Tensor<[3, 4], f64> = zeros([5, 6])` produces compile error
- [ ] All M1-M9 examples still compile and run with identical output
- [ ] `cargo test --workspace` passes
- [ ] `--dump-types` shows human-readable tensor shapes

---

## File Summary

| File | Action | Task |
|------|--------|------|
| `crates/nsl-semantic/src/checker.rs` | Modify | T1, T3, T4, T6 |
| `crates/nsl-semantic/src/types.rs` | Modify | T2, T8 |
| `crates/nsl-semantic/src/shapes.rs` | Modify | T6 |
| `crates/nsl-cli/src/main.rs` | Modify | T8 |
| `examples/m10_shape_check.nsl` | Create | T1, T2, T3, T4 |
| `examples/m10_shape_errors.nsl` | Create | T1, T2, T4 |
| `examples/m10_symbolic_dims.nsl` | Create | T5 |
| `examples/m10_fn_shape_errors.nsl` | Create | T5 |
| `tests/expected/m10_shape_check.txt` | Create | T7 |

**Estimated scope:** ~200 lines of Rust changes (checker.rs: ~120, types.rs: ~40, shapes.rs: ~30, main.rs: ~10) + test programs.
