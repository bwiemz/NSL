# FBIP (Functional But In-Place) Tensor Mutation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate millions of unnecessary tensor allocations by detecting uniquely-owned tensors at runtime (`refcount == 1`) and mutating them in-place instead of allocating new output tensors. NSL already has the critical infrastructure — `NslTensor.refcount: AtomicI64` — but every operation currently allocates fresh output regardless of ownership. This is the single highest-impact optimization for training workloads.

**Architecture:** A two-level approach: (1) runtime reuse checks inserted into every unary and binary tensor operation (`if refcount == 1 && owns_data && !is_view → mutate in-place`), (2) optional compile-time reuse analysis in the ownership checker that elides refcount checks when uniqueness is statically provable. Phase 1 is runtime-only (safe, simple). Phase 2 adds static elision (faster, requires M38a).

**Tech Stack:** Rust, existing `nsl-runtime/src/tensor/` (arithmetic.rs, activation.rs, mod.rs)

**Research Basis:** Type Systems & Safety notebook describes Koka's Perceus reference counting with reuse analysis. NSL's approach is simpler — we already have atomic refcounting; we just need the conditional mutation paths.

---

## Background

### Current Allocation Pattern (Every Op)

Every tensor operation follows this pattern:
```rust
pub extern "C" fn nsl_tensor_relu(ptr: i64) -> i64 {
    let t = as_tensor(ptr);
    let data = t.contiguous_data();
    let new_data = checked_alloc(t.len * sizeof(f64));  // ALWAYS allocates
    for i in 0..t.len {
        new_data[i] = f64::max(data[i], 0.0);
    }
    let result = NslTensor::new(new_data, t.shape.clone(), refcount: 1);
    Box::into_raw(result) as i64
}
```

### FBIP Pattern (Reuse When Safe)

```rust
pub extern "C" fn nsl_tensor_relu(ptr: i64) -> i64 {
    let t = as_tensor(ptr);
    if t.can_mutate_inplace() {     // refcount==1 && owns_data && !is_view
        // FAST PATH: mutate in-place, return same pointer
        let data = t.data as *mut f64;
        for i in 0..t.len { data[i] = f64::max(data[i], 0.0); }
        ptr  // return same tensor
    } else {
        // SLOW PATH: allocate new (current behavior)
        let new_data = checked_alloc(t.len * sizeof(f64));
        // ... copy with transform ...
    }
}
```

### Why This Is Safe

1. `refcount == 1` means no other code holds a reference to this tensor
2. `owns_data == 1` means this tensor owns its data buffer (not a view)
3. `data_owner == 0` means no other tensor's data depends on this buffer
4. Autodiff tape bumps refcount to 2 before saving → FBIP naturally falls through to allocation path when tape is recording
5. `@shared` tensors have refcount ≥ 2 → always fall through to allocation

### Existing In-Place Operations

These already exist but are **unsafe** (no refcount check):
- `nsl_tensor_copy_data()` — shallow buffer copy
- `nsl_tensor_add_inplace()` — `dst += src`
- `nsl_tensor_zero_inplace()` — `dst[:] = 0`

FBIP makes the refcount-safe version the default path for ALL operations.

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/nsl-runtime/src/tensor/mod.rs` | Add `can_mutate_inplace()` method, `try_reuse()` helper |
| `crates/nsl-runtime/src/tensor/activation.rs` | Add FBIP paths to relu, gelu, silu, sigmoid, tanh, softmax, exp, log, sqrt, abs, neg |
| `crates/nsl-runtime/src/tensor/arithmetic.rs` | Add FBIP paths to add, sub, mul, div (reuse left operand when unique) |
| `crates/nsl-runtime/src/tensor/shape_ops.rs` | Add FBIP paths to clone (becomes no-op when unique) |
| Tests | Verify in-place mutation when refcount==1, allocation when refcount>1 |

---

## Phase 1: Runtime Reuse Checks

### Task 1: Add `can_mutate_inplace()` Method

- [ ] **1.1** Add method to NslTensor:
```rust
impl NslTensor {
    /// Returns true if this tensor can be safely mutated in-place.
    /// Requires: sole owner (refcount==1), owns data (not a view),
    /// contiguous layout, and CPU device (GPU has separate path).
    #[inline]
    pub fn can_mutate_inplace(&self) -> bool {
        self.refcount.load(Ordering::Acquire) == 1
            && self.owns_data == 1
            && self.data_owner == 0
    }
}
```

- [ ] **1.2** Add `can_mutate_inplace_gpu()` for GPU tensors (same logic, different dtype handling).

- [ ] **1.3** Test: tensor with refcount=1 returns true; tensor with refcount=2 returns false; view tensor returns false.

### Task 2: FBIP for Unary Activation Functions

- [ ] **2.1** Add FBIP to `nsl_tensor_relu()`:
```rust
pub extern "C" fn nsl_tensor_relu(ptr: i64) -> i64 {
    let t = unsafe { &mut *(ptr as *mut NslTensor) };
    if t.can_mutate_inplace() {
        let data = t.data as *mut f64;
        let len = t.len as usize;
        for i in 0..len {
            let v = unsafe { *data.add(i) };
            if v < 0.0 { unsafe { *data.add(i) = 0.0 }; }
        }
        return ptr; // return same tensor — zero allocation
    }
    // ... existing allocation path unchanged ...
}
```

- [ ] **2.2** Apply same pattern to: `neg`, `abs`, `exp`, `log`, `sqrt`, `sigmoid`, `tanh`, `gelu`, `silu`.

- [ ] **2.3** For each: the in-place path modifies `data[i]` directly and returns the same pointer.

- [ ] **2.4** Test for each activation: create tensor with refcount=1, apply activation, verify same pointer returned and values correct.

### Task 3: FBIP for Binary Arithmetic Operations

- [ ] **3.1** Add FBIP to `nsl_tensor_add()` — reuse left operand:
```rust
pub extern "C" fn nsl_tensor_add(a_ptr: i64, b_ptr: i64) -> i64 {
    let a = unsafe { &mut *(a_ptr as *mut NslTensor) };
    let b = as_tensor(b_ptr);

    // Can reuse 'a' if: same shape (no broadcast), a is unique, both contiguous
    if a.can_mutate_inplace() && a.shape_eq(b) && a.is_contiguous() && b.is_contiguous() {
        let a_data = a.data as *mut f64;
        let b_data = b.contiguous_data();
        for i in 0..a.len as usize {
            unsafe { *a_data.add(i) += *b_data.add(i) };
        }
        return a_ptr;
    }
    // ... existing allocation path with broadcasting ...
}
```

- [ ] **3.2** Apply to `sub`, `mul`, `div` — all reuse left operand when shapes match.

- [ ] **3.3** For broadcasting cases: cannot reuse (output shape differs from input). Fall through to allocation.

- [ ] **3.4** Test: `a + b` where `a` has refcount=1 and same shape → returns `a_ptr`; where `a` has refcount=2 → returns new pointer.

### Task 4: FBIP for Clone (Becomes No-Op)

- [ ] **4.1** `nsl_tensor_clone()` with FBIP:
```rust
pub extern "C" fn nsl_tensor_clone(ptr: i64) -> i64 {
    let t = as_tensor(ptr);
    if t.can_mutate_inplace() {
        // Sole owner — clone is unnecessary, return same tensor
        return ptr;
    }
    // ... existing deep copy ...
}
```

- [ ] **4.2** This is the highest-impact single change — `clone()` is called frequently in training loops.

### Task 5: Autodiff Tape Safety Verification

- [ ] **5.1** Verify tape recording makes FBIP safe:
  - When autodiff is recording, tape ops bump refcount to 2+ on saved tensors
  - FBIP check (`refcount == 1`) naturally fails → allocation path taken
  - No tensor saved on the tape can be mutated in-place

- [ ] **5.2** Test: inside `grad()` scope, verify relu allocates new tensor (refcount > 1 due to tape).

- [ ] **5.3** Test: outside `grad()` scope (inference), verify relu reuses tensor.

### Task 6: GPU In-Place Paths

- [ ] **6.1** For GPU tensors: same refcount check, but in-place kernel writes to same device buffer:
```rust
if t.can_mutate_inplace() && t.device > 0 {
    // Launch GPU kernel with input buffer as both input and output
    cuda_launch_inplace_relu(t.data, t.len);
    return ptr;
}
```

- [ ] **6.2** GPU in-place requires: same device, contiguous, sole owner.

- [ ] **6.3** Test: GPU tensor with refcount=1 → in-place kernel; refcount=2 → new allocation.

### Task 7: Metrics & Reporting

- [ ] **7.1** Add optional FBIP counter (behind `--trace` flag):
```rust
static FBIP_REUSE_COUNT: AtomicU64 = AtomicU64::new(0);
static FBIP_ALLOC_COUNT: AtomicU64 = AtomicU64::new(0);
```

- [ ] **7.2** At program exit, report: `"FBIP: {reuse}/{total} operations reused in-place ({pct}%)"`.

- [ ] **7.3** This helps quantify the memory savings in practice.

---

## Phase 2: Static Reuse Analysis (Future, Requires M38a)

### Task 8: Compile-Time Uniqueness Proofs

- [ ] **8.1** When the ownership checker (M38a) proves a tensor is `Owned` and will be consumed by the next operation, the codegen can:
  - Skip the runtime `refcount == 1` check entirely
  - Emit a direct call to the in-place variant
  - This saves one atomic load per operation

- [ ] **8.2** Emit `nsl_tensor_relu_inplace(ptr)` instead of `nsl_tensor_relu(ptr)` when statically proven safe.

- [ ] **8.3** This is a codegen optimization, not a correctness change — Phase 1 runtime checks are always safe.

---

## Expected Impact

| Workload | Current Allocs | With FBIP | Reduction |
|----------|---------------|-----------|-----------|
| Forward pass (inference) | 1 per op | ~0 per op (all unique) | ~100% |
| Forward pass (training) | 1 per op | 1 per op (tape bumps refcount) | 0% |
| Backward pass | 1 per op | ~50% reused (grads are unique) | ~50% |
| Optimizer step | 1 per param update | ~0 (params are unique after backward) | ~100% |

The biggest win is inference: every tensor operation becomes zero-allocation.

---

## Effort Estimate

- Task 1 (can_mutate_inplace): 0.25 days
- Task 2 (unary activations): 1 day
- Task 3 (binary arithmetic): 1 day
- Task 4 (clone no-op): 0.25 days
- Task 5 (autodiff verification): 0.5 days
- Task 6 (GPU paths): 0.5 days
- Task 7 (metrics): 0.25 days
- Total Phase 1: **4 days**
- Phase 2 (static analysis): 2-3 days (after M38a matures)
