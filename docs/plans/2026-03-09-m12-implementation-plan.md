# M12: Autodiff Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement tape-based automatic differentiation so `grad` blocks compute real gradients for tensor parameters.

**Architecture:** Global tape records tensor operations at runtime. When code enters a `grad` block, recording starts. All tensor ops (add, mul, matmul, etc.) check a thread-local flag and record themselves on the tape. After the forward pass, the backward engine walks the tape in reverse applying the chain rule. This approach means functions called from within a grad block automatically participate in gradient tracking (like PyTorch's autograd).

**Tech Stack:** Rust (runtime autodiff engine), Cranelift 0.116 (codegen), NSL parser/semantic/codegen crates

---

## Current State

**What exists:**
- Lexer: `grad` keyword (`TokenKind::Grad`), `@` decorator token (`TokenKind::At`)
- Parser: `parse_grad_block_stmt()` in `crates/nsl-parser/src/block.rs:154-174` — parses `grad(targets): body`
- AST: `GradBlock { targets: Expr, body: Block, span }` in `crates/nsl-ast/src/block.rs:42-46`
- AST: `Decorator { name, args, span }` in `crates/nsl-ast/src/decl.rs:96-100`
- Semantic: Minimal check in `crates/nsl-semantic/src/checker.rs:271-274` — just walks targets and body
- Types: `Type::Param { shape, dtype }` and `Type::Buffer { shape, dtype }` in `crates/nsl-semantic/src/types.rs:42-49`
- Codegen: `StmtKind::GradBlock(_)` is a **no-op** in `crates/nsl-codegen/src/stmt.rs:175`
- Runtime: No autodiff infrastructure. 27+ tensor ops in `crates/nsl-runtime/src/tensor.rs` (add, sub, mul, div, matmul, neg, mul_scalar, add_scalar, sum, mean, etc.)
- Runtime: `NslTensor` struct has `refcount` field for sharing
- M11 complete: models with fields, methods, forward dispatch all work

**What's missing (what M12 adds):**
1. Tape data structure + backward engine in Rust runtime
2. Tape recording in existing tensor ops (global flag approach)
3. `.sum()` / `.mean()` universally return 0-d scalar tensors (not f64) + `.item()` method
4. Parser support for `let loss, grads = grad(targets): body`
5. Semantic validation of grad blocks
6. Codegen for grad blocks (start tape, compile body, backward, bind results)
7. `@no_grad` decorator support
8. Update existing code that expects `.sum()` / `.mean()` to return f64

**What's deferred to later milestones:**
- `@checkpoint` (activation checkpointing — M13+)
- `@backward` / `@custom_vjp` (custom gradient definitions — M13+)
- Second-order gradients (nested grad blocks — M13+)
- `model.params()` method (returns all Param tensors — M13)
- Gradient clipping (can be done manually or in M13 optimizer stdlib)
- Inline grad expressions `grad(expr, wrt=target)` — M13+
- Basic REPL (separate sub-milestone, requires cranelift-jit dependency)
- In-place tensor mutation guards (version counters / copy-on-write when refcount > 1)

---

## Architecture: Global Tape Approach

```
Grad block execution flow:
1. Evaluate targets → list of param tensor ptrs
2. Call nsl_tape_start(param_list) → sets recording flag, registers params
3. Execute body (normal compilation — NO compile-time switching)
   └─ All tensor ops check recording flag → record on tape if active
   └─ Functions called from body also auto-record (transparent)
4. Last expression = loss scalar tensor (since sum/mean always return tensors)
5. Call nsl_tape_backward(loss_tensor, param_list) → gradient tensors
6. Call nsl_tape_stop() → clears flag, frees tape + decrements saved tensor refcounts
7. Extract loss f64 via .item(), bind loss (f64) and grads to variables

Tape entry = (op_type, input_ptrs, output_ptr, saved_tensors)
Backward = walk tape in reverse, apply chain rule, accumulate grads
```

**Key design decisions:**

1. **Global tape** (not separate AD function set): tensor ops check a thread-local flag. This means any function called from a grad block automatically participates. No compile-time flag needed.

2. **Refcount-based tensor saving**: tape increments refcount of tensors it saves for backward, preventing premature deallocation. **Cleanup is in `nsl_tape_stop`** (not `nsl_tape_backward`), guaranteeing cleanup even if backward is never called (error, early return, etc.).

3. **`.sum()` and `.mean()` ALWAYS return 0-d scalar tensors** (i64 pointer, not f64). This is critical for the AOT compiler — since functions are compiled once, we cannot switch behavior based on whether the call site is inside a grad block. Users extract f64 with `.item()`. This matches the spec (Section 3, Example 4 uses `loss.item()`).

4. **In-place mutation**: Not guarded in M12. TODO: when tensor element assignment (`y[0] = 5.0`) is added, if the tensor has refcount > 1 (saved on tape), either ban mutation or copy-on-write.

---

## Task 1: Runtime — Tape Data Structure

**Files:**
- Create: `crates/nsl-runtime/src/autodiff.rs`
- Modify: `crates/nsl-runtime/src/lib.rs` (add `mod autodiff;`)

**Step 1: Create the autodiff module with tape types**

Create `crates/nsl-runtime/src/autodiff.rs`:

```rust
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

/// Operations recorded on the autodiff tape.
#[derive(Clone, Debug)]
pub enum TapeOp {
    Add { a: i64, b: i64, out: i64 },
    Sub { a: i64, b: i64, out: i64 },
    Mul { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64 },
    Div { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64 },
    MatMul { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64 },
    Neg { a: i64, out: i64 },
    MulScalar { a: i64, scalar: f64, out: i64 },
    AddScalar { a: i64, out: i64 },
    SumReduce { a: i64, out: i64 },
    MeanReduce { a: i64, out: i64, num_elements: i64 },
}

/// The autodiff tape. Records operations during forward pass.
struct Tape {
    ops: Vec<TapeOp>,
    param_set: HashSet<i64>,   // tensor ptrs that are grad targets
    recording: bool,
    pause_depth: i32,          // for nested @no_grad
}

impl Tape {
    fn new() -> Self {
        Tape {
            ops: Vec::new(),
            param_set: HashSet::new(),
            recording: false,
            pause_depth: 0,
        }
    }

    fn is_recording(&self) -> bool {
        self.recording && self.pause_depth == 0
    }
}

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
}

/// Check if tape is currently recording. Called by tensor ops.
pub fn is_recording() -> bool {
    TAPE.with(|t| t.borrow().is_recording())
}

/// Record an operation on the tape (if recording).
pub fn maybe_record(op: TapeOp) {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        if tape.is_recording() {
            tape.ops.push(op);
        }
    });
}
```

**Step 2: Add tape start/stop/pause extern "C" functions**

Add to the same file:

```rust
use crate::list::NslList;
use crate::tensor::NslTensor;

/// Start tape recording. param_list is an NslList of tensor ptrs to compute gradients for.
#[no_mangle]
pub extern "C" fn nsl_tape_start(param_list: i64) {
    let list = NslList::from_ptr(param_list);
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        tape.ops.clear();
        tape.param_set.clear();
        for i in 0..list.len as usize {
            let ptr = unsafe { *list.data.add(i) };
            tape.param_set.insert(ptr);
        }
        tape.recording = true;
        tape.pause_depth = 0;
    });
}

/// Stop tape recording and clean up.
/// IMPORTANT: This handles saved tensor refcount cleanup, guaranteeing no leaks
/// even if nsl_tape_backward was never called (error, early return, etc.).
#[no_mangle]
pub extern "C" fn nsl_tape_stop() {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        tape.recording = false;
        // Decrement refcounts of saved tensors before clearing ops
        for op in &tape.ops {
            match op {
                TapeOp::Mul { saved_a, saved_b, .. }
                | TapeOp::Div { saved_a, saved_b, .. }
                | TapeOp::MatMul { saved_a, saved_b, .. } => {
                    NslTensor::from_ptr(*saved_a).refcount -= 1;
                    NslTensor::from_ptr(*saved_b).refcount -= 1;
                }
                _ => {}
            }
        }
        tape.ops.clear();
        tape.param_set.clear();
    });
}

/// Pause recording (for @no_grad). Supports nesting.
#[no_mangle]
pub extern "C" fn nsl_tape_pause() {
    TAPE.with(|t| {
        t.borrow_mut().pause_depth += 1;
    });
}

/// Resume recording (after @no_grad).
#[no_mangle]
pub extern "C" fn nsl_tape_resume() {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        if tape.pause_depth > 0 {
            tape.pause_depth -= 1;
        }
    });
}
```

**Step 3: Wire module into lib.rs**

In `crates/nsl-runtime/src/lib.rs`, add `pub mod autodiff;`.

**Step 4: Verify it compiles**

Run: `cargo build -p nsl-runtime`
Expected: Compiles successfully

---

## Task 2: Runtime — Change sum/mean to Return Scalar Tensors + Add .item() + Tape Recording

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`

**Goal:** Make `nsl_tensor_sum` and `nsl_tensor_mean` return 0-dimensional scalar tensors (i64) instead of f64. Add `nsl_tensor_item` to extract f64 from a scalar tensor. Add tape recording to all tensor ops. This is critical for the AOT compiler — since functions are compiled once, sum/mean must always return the same type regardless of call context.

**IMPORTANT — Breaking change:** This changes the return type of `nsl_tensor_sum` and `nsl_tensor_mean` from f64 to i64. All callers in the codegen must be updated (Task 6).

**Step 1: Add scalar tensor creation helper**

Add to `tensor.rs`:

```rust
/// Create a 0-d tensor with a single scalar value.
fn create_scalar_tensor(value: f64) -> i64 {
    let data = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
    unsafe { *data = value };

    let tensor = Box::new(NslTensor {
        data,
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        ndim: 0,
        len: 1,
        refcount: 1,
    });
    Box::into_raw(tensor) as i64
}
```

**Step 2: Change nsl_tensor_sum to return scalar tensor**

```rust
// BEFORE:
// pub extern "C" fn nsl_tensor_sum(tensor_ptr: i64) -> f64

// AFTER:
/// Sum all elements, return a 0-d scalar tensor.
#[no_mangle]
pub extern "C" fn nsl_tensor_sum(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let mut total = 0.0;
    for i in 0..tensor.len as usize {
        total += unsafe { *tensor.data.add(i) };
    }
    let result = create_scalar_tensor(total);
    if crate::autodiff::is_recording() {
        crate::autodiff::maybe_record(crate::autodiff::TapeOp::SumReduce {
            a: tensor_ptr, out: result,
        });
    }
    result
}
```

**Step 3: Change nsl_tensor_mean to return scalar tensor**

```rust
// BEFORE:
// pub extern "C" fn nsl_tensor_mean(tensor_ptr: i64) -> f64

// AFTER:
/// Mean of all elements, return a 0-d scalar tensor.
#[no_mangle]
pub extern "C" fn nsl_tensor_mean(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let total = if tensor.len == 0 {
        0.0
    } else {
        let mut s = 0.0;
        for i in 0..tensor.len as usize {
            s += unsafe { *tensor.data.add(i) };
        }
        s / tensor.len as f64
    };
    let num_elements = tensor.len;
    let result = create_scalar_tensor(total);
    if crate::autodiff::is_recording() {
        crate::autodiff::maybe_record(crate::autodiff::TapeOp::MeanReduce {
            a: tensor_ptr, out: result, num_elements,
        });
    }
    result
}
```

**Step 4: Add .item() — extract f64 from scalar tensor**

```rust
/// Extract the scalar f64 value from a 0-d (or 1-element) tensor.
/// This is the NSL equivalent of PyTorch's .item().
#[no_mangle]
pub extern "C" fn nsl_tensor_item(tensor_ptr: i64) -> f64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.len != 1 {
        eprintln!(
            "nsl: .item() requires a scalar tensor (got {} elements)",
            tensor.len
        );
        std::process::abort();
    }
    unsafe { *tensor.data }
}
```

**Step 5: Add tape recording to other tensor ops**

Add `use crate::autodiff;` at the top of `tensor.rs`.

For ops that DON'T need saved tensors (add, sub, neg, add_scalar):

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_add(a: i64, b: i64) -> i64 {
    let result = tensor_elementwise_op(a, b, |x, y| x + y);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Add { a, b, out: result });
    }
    result
}
```

For ops that DO need saved tensors (mul, div, matmul) — increment refcount:

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_mul(a: i64, b: i64) -> i64 {
    let result = tensor_elementwise_op(a, b, |x, y| x * y);
    if autodiff::is_recording() {
        NslTensor::from_ptr(a).refcount += 1;
        NslTensor::from_ptr(b).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Mul {
            a, b, out: result, saved_a: a, saved_b: b,
        });
    }
    result
}
```

Apply tape recording to ALL these ops:
- `nsl_tensor_add` → TapeOp::Add (no saves)
- `nsl_tensor_sub` → TapeOp::Sub (no saves)
- `nsl_tensor_mul` → TapeOp::Mul (save a, b — increment refcounts)
- `nsl_tensor_div` → TapeOp::Div (save a, b — increment refcounts)
- `nsl_tensor_matmul` → TapeOp::MatMul (save a, b — increment refcounts)
- `nsl_tensor_neg` → TapeOp::Neg (no saves)
- `nsl_tensor_mul_scalar` → TapeOp::MulScalar (no tensor saves, scalar value stored in TapeOp)
- `nsl_tensor_add_scalar` → TapeOp::AddScalar (no saves)

**Step 6: Update nsl_tensor_free for null shape/strides**

Since scalar tensors have null shape and strides pointers, ensure `nsl_tensor_free` handles this:

```rust
// In nsl_tensor_free, guard against null pointers:
if tensor.refcount <= 0 {
    let data_size = (tensor.len as usize) * std::mem::size_of::<f64>();
    unsafe {
        checked_free(tensor.data as *mut u8, data_size);
        if !tensor.shape.is_null() {
            let shape_size = (tensor.ndim as usize) * std::mem::size_of::<i64>();
            checked_free(tensor.shape as *mut u8, shape_size);
        }
        if !tensor.strides.is_null() {
            let strides_size = (tensor.ndim as usize) * std::mem::size_of::<i64>();
            checked_free(tensor.strides as *mut u8, strides_size);
        }
        drop(Box::from_raw(tensor as *mut NslTensor));
    }
}
```

**Step 7: Verify it compiles**

Run: `cargo build -p nsl-runtime`
Expected: Compiles successfully

---

## Task 3: Runtime — Backward Engine

**Files:**
- Modify: `crates/nsl-runtime/src/autodiff.rs`

**Goal:** Implement the backward pass that walks the tape in reverse, applying the chain rule to compute gradients for each parameter tensor.

**Step 1: Implement helper functions for gradient operations**

Add to `autodiff.rs`:

```rust
use crate::tensor::{
    nsl_tensor_add as tensor_add,
    nsl_tensor_neg as tensor_neg,
    nsl_tensor_mul as tensor_mul,
    nsl_tensor_div as tensor_div,
    nsl_tensor_matmul as tensor_matmul,
    nsl_tensor_transpose as tensor_transpose,
    nsl_tensor_mul_scalar as tensor_mul_scalar,
    nsl_tensor_clone as tensor_clone,
    nsl_tensor_ones as tensor_ones,
    nsl_tensor_shape as tensor_shape,
    nsl_tensor_free as tensor_free,
};

/// Create a tensor of ones with the same shape as the given tensor.
fn ones_like(tensor_ptr: i64) -> i64 {
    let shape_list = tensor_shape(tensor_ptr);
    tensor_ones(shape_list)
}

/// Accumulate gradient: grads[key] += grad_tensor.
/// If grads[key] doesn't exist, set it to grad_tensor.
/// If it does exist, add grad_tensor to it (and free the old sum and the grad_tensor).
fn accumulate_grad(grads: &mut HashMap<i64, i64>, key: i64, grad_tensor: i64) {
    if let Some(existing) = grads.get(&key) {
        let old = *existing;
        // Pause recording to avoid taping gradient computation ops
        TAPE.with(|t| t.borrow_mut().pause_depth += 1);
        let summed = tensor_add(old, grad_tensor);
        TAPE.with(|t| t.borrow_mut().pause_depth -= 1);
        tensor_free(old);
        tensor_free(grad_tensor);
        grads.insert(key, summed);
    } else {
        grads.insert(key, grad_tensor);
    }
}
```

**Step 2: Implement the backward pass**

```rust
/// Run backward pass. loss_ptr is the scalar tensor (loss). param_list is an NslList of param ptrs.
/// Returns an NslList of gradient tensors (one per param, same order as param_list).
#[no_mangle]
pub extern "C" fn nsl_tape_backward(loss_ptr: i64, param_list: i64) -> i64 {
    let params_nsl = NslList::from_ptr(param_list);
    let param_ptrs: Vec<i64> = (0..params_nsl.len as usize)
        .map(|i| unsafe { *params_nsl.data.add(i) })
        .collect();

    // Pause recording during backward (don't tape gradient computation)
    TAPE.with(|t| t.borrow_mut().pause_depth += 1);

    let ops = TAPE.with(|t| t.borrow().ops.clone());

    // grad_map: tensor_ptr -> gradient tensor ptr
    let mut grad_map: HashMap<i64, i64> = HashMap::new();

    // Seed: gradient of loss w.r.t. itself = ones_like(loss)
    let seed = ones_like(loss_ptr);
    grad_map.insert(loss_ptr, seed);

    // Walk tape in reverse
    for op in ops.iter().rev() {
        match op {
            TapeOp::Add { a, b, out } => {
                if let Some(&g) = grad_map.get(out) {
                    accumulate_grad(&mut grad_map, *a, tensor_clone(g));
                    accumulate_grad(&mut grad_map, *b, tensor_clone(g));
                }
            }
            TapeOp::Sub { a, b, out } => {
                if let Some(&g) = grad_map.get(out) {
                    accumulate_grad(&mut grad_map, *a, tensor_clone(g));
                    let neg_g = tensor_neg(tensor_clone(g));
                    accumulate_grad(&mut grad_map, *b, neg_g);
                }
            }
            TapeOp::Mul { a, b, out, saved_a, saved_b } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/da (a*b) = g * b, d/db (a*b) = g * a
                    let grad_a = tensor_mul(tensor_clone(g), *saved_b);
                    let grad_b = tensor_mul(tensor_clone(g), *saved_a);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::Div { a, b, out, saved_a, saved_b } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/da (a/b) = g/b
                    let grad_a = tensor_div(tensor_clone(g), *saved_b);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                    // d/db (a/b) = -g*a/b^2
                    let neg_g = tensor_neg(tensor_clone(g));
                    let neg_ga = tensor_mul(neg_g, *saved_a);
                    let b_sq = tensor_mul(*saved_b, *saved_b);
                    let grad_b = tensor_div(neg_ga, b_sq);
                    tensor_free(neg_g);
                    tensor_free(neg_ga);
                    tensor_free(b_sq);
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::MatMul { a, b, out, saved_a, saved_b } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/dA (A@B) = G @ B^T
                    // d/dB (A@B) = A^T @ G
                    let b_t = tensor_transpose(*saved_b, 0, 1);
                    let a_t = tensor_transpose(*saved_a, 0, 1);
                    let grad_a = tensor_matmul(tensor_clone(g), b_t);
                    let grad_b = tensor_matmul(a_t, tensor_clone(g));
                    tensor_free(b_t);
                    tensor_free(a_t);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::Neg { a, out } => {
                if let Some(&g) = grad_map.get(out) {
                    accumulate_grad(&mut grad_map, *a, tensor_neg(tensor_clone(g)));
                }
            }
            TapeOp::MulScalar { a, scalar, out } => {
                if let Some(&g) = grad_map.get(out) {
                    accumulate_grad(&mut grad_map, *a, tensor_mul_scalar(tensor_clone(g), *scalar));
                }
            }
            TapeOp::AddScalar { a, out } => {
                if let Some(&g) = grad_map.get(out) {
                    accumulate_grad(&mut grad_map, *a, tensor_clone(g));
                }
            }
            TapeOp::SumReduce { a, out } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/da sum(a) = ones_like(a) * scalar_grad
                    // g is a 0-d tensor; broadcast its value to shape of a
                    let scalar_val = unsafe { *NslTensor::from_ptr(g).data };
                    let grad_a = ones_like(*a);
                    let grad_a = tensor_mul_scalar(grad_a, scalar_val);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::MeanReduce { a, out, num_elements } => {
                if let Some(&g) = grad_map.get(out) {
                    let scalar_val = unsafe { *NslTensor::from_ptr(g).data };
                    let grad_a = ones_like(*a);
                    let grad_a = tensor_mul_scalar(grad_a, scalar_val / (*num_elements as f64));
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
        }
    }

    // Resume recording
    TAPE.with(|t| t.borrow_mut().pause_depth -= 1);

    // Build result list: one gradient per param (in same order as param_list)
    let result_list = crate::list::nsl_list_new();
    for ptr in &param_ptrs {
        if let Some(&grad) = grad_map.get(ptr) {
            crate::list::nsl_list_push(result_list, grad);
        } else {
            // No gradient computed for this param — return zeros_like
            let shape_list = tensor_shape(*ptr);
            let zeros = crate::tensor::nsl_tensor_zeros(shape_list);
            crate::list::nsl_list_push(result_list, zeros);
        }
    }

    // NOTE: Saved tensor refcount cleanup is handled by nsl_tape_stop(),
    // NOT here. This ensures cleanup even if backward is never called.

    result_list
}
```

**Important implementation notes:**
- The `TapeOp` enum must be `pub` so `tensor.rs` can construct variants.
- `NslTensor::from_ptr` is already `pub` on the impl.
- During backward, `pause_depth` is incremented so gradient computation ops don't get recorded on the tape.
- `tensor_clone(g)` is used when a gradient is consumed multiple times (e.g., Add gives g to both inputs).
- The calls to `tensor_mul`, `tensor_div`, etc. in backward won't record because `pause_depth > 0`.
- TODO (in-place mutation): If a saved tensor was mutated between forward and backward, gradients will be wrong. For M12 this is not guarded — add version counters or copy-on-write in a future milestone.

**Step 2: Verify it compiles**

Run: `cargo build -p nsl-runtime`
Expected: Compiles successfully

**Step 3: Run unit tests**

Run: `cargo test --workspace`
Expected: All existing unit tests still pass. Integration tests may fail due to sum/mean return type change — fixed in Task 6.

---

## Task 4: AST + Parser — Grad Block Output Bindings

**Files:**
- Modify: `crates/nsl-ast/src/block.rs:42-46` (GradBlock struct)
- Modify: `crates/nsl-parser/src/stmt.rs` (let statement parsing)
- Modify: `crates/nsl-parser/src/block.rs:154-174` (parse_grad_block_stmt)

**Goal:** Support `let loss, grads = grad(targets): body` syntax by adding output bindings to GradBlock.

**Step 1: Modify GradBlock AST to include output bindings**

In `crates/nsl-ast/src/block.rs`, change:

```rust
// BEFORE:
pub struct GradBlock {
    pub targets: Expr,
    pub body: Block,
    pub span: Span,
}

// AFTER:
pub struct GradBlock {
    pub outputs: Option<crate::pattern::Pattern>,  // `let loss, grads = ...` binding pattern
    pub targets: Expr,
    pub body: Block,
    pub span: Span,
}
```

**Step 2: Update parse_grad_block_stmt for standalone grad blocks**

In `crates/nsl-parser/src/block.rs`, update `parse_grad_block_stmt` to set `outputs: None`:

```rust
pub fn parse_grad_block_stmt(p: &mut Parser) -> Stmt {
    let start = p.current_span();
    p.advance(); // consume 'grad'

    p.expect(&TokenKind::LeftParen);
    let targets = parse_expr(p);
    p.expect(&TokenKind::RightParen);
    p.expect(&TokenKind::Colon);
    let body = p.parse_block();

    let span = start.merge(body.span);
    Stmt {
        kind: StmtKind::GradBlock(GradBlock {
            outputs: None,
            targets,
            body: body.clone(),
            span,
        }),
        span,
        id: p.next_node_id(),
    }
}
```

**Step 3: Handle `let <pattern> = grad(targets):` in VarDecl parsing**

Find where `VarDecl` (let/const statements) are parsed in `crates/nsl-parser/src/stmt.rs`. When parsing the value expression after `=`, check if the next token is `grad`. If so, parse as a GradBlock with output bindings instead of a normal VarDecl.

The exact location depends on the parser structure. Look for where `StmtKind::VarDecl` is constructed, specifically the path where `value: Some(expr)` is set. The logic should be approximately:

```rust
// In the let/const parsing logic, after parsing the pattern and `=`:
if p.at(&TokenKind::Grad) {
    // Parse as grad block with bindings
    p.advance(); // consume 'grad'
    p.expect(&TokenKind::LeftParen);
    let targets = parse_expr(p);
    p.expect(&TokenKind::RightParen);
    p.expect(&TokenKind::Colon);
    let body = p.parse_block();
    let span = start.merge(body.span);
    return Stmt {
        kind: StmtKind::GradBlock(GradBlock {
            outputs: Some(pattern),  // the let binding pattern
            targets,
            body: body.clone(),
            span,
        }),
        span,
        id: p.next_node_id(),
    };
}
// Otherwise, parse as normal expression value for VarDecl
```

**Step 4: Verify it compiles and tests pass**

Run: `cargo test --workspace`
Expected: All tests pass

---

## Task 5: Semantic Checker — Grad Block Validation

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs:271-274`

**Goal:** Improve grad block type checking: validate targets, check body, declare output bindings.

**Step 1: Enhance GradBlock checking**

Replace the minimal GradBlock check in `checker.rs:271-274` with:

```rust
StmtKind::GradBlock(grad) => {
    // Check targets expression (should evaluate to tensor(s))
    self.check_expr(&grad.targets);

    // Check body in a new block scope
    self.check_block(&grad.body, ScopeKind::Block);

    // If there are output bindings, declare them in the enclosing scope
    if let Some(ref pattern) = grad.outputs {
        match &pattern.kind {
            PatternKind::Tuple(pats) => {
                for p in pats {
                    if let PatternKind::Ident(sym) = &p.kind {
                        self.declare_symbol(*sym, Type::Unknown, p.span, false, true);
                    }
                }
            }
            PatternKind::Ident(sym) => {
                self.declare_symbol(*sym, Type::Unknown, pattern.span, false, true);
            }
            _ => {}
        }
    }
}
```

Check if `PatternKind` needs to be imported from `nsl_ast::pattern::PatternKind`.

**Step 2: Verify it compiles and tests pass**

Run: `cargo test --workspace`
Expected: All tests pass

---

## Task 6: Codegen — Declare AD Runtime Functions + Update sum/mean Signatures

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr.rs` (update sum/mean call sites + add .item())

**Goal:** Add the new autodiff runtime functions to the builtins declaration list. Update `nsl_tensor_sum` and `nsl_tensor_mean` signatures from returning F64 to I64. Add `.item()` method support. Fix all call sites.

**Step 1: Update sum/mean return types in RUNTIME_FUNCTIONS**

In `builtins.rs`, find and change:

```rust
// BEFORE:
("nsl_tensor_sum", &[types::I64], Some(types::F64)),
("nsl_tensor_mean", &[types::I64], Some(types::F64)),

// AFTER:
("nsl_tensor_sum", &[types::I64], Some(types::I64)),   // now returns scalar tensor ptr
("nsl_tensor_mean", &[types::I64], Some(types::I64)),   // now returns scalar tensor ptr
```

**Step 2: Add AD + .item() functions to RUNTIME_FUNCTIONS**

```rust
// Tensor .item() — extract scalar f64
("nsl_tensor_item", &[types::I64], Some(types::F64)),
// Autodiff tape management
("nsl_tape_start", &[types::I64], None),
("nsl_tape_stop", &[], None),
("nsl_tape_backward", &[types::I64, types::I64], Some(types::I64)),
("nsl_tape_pause", &[], None),
("nsl_tape_resume", &[], None),
```

**Step 3: Update codegen for .sum() and .mean()**

Search `expr.rs` for where `.sum()` and `.mean()` are compiled (look for string matches on `"sum"`, `"mean"` in the tensor method dispatch). These calls now return I64 (scalar tensor ptr) instead of F64. Update the Cranelift return type handling:

```rust
// Where .sum() is compiled:
// The call to nsl_tensor_sum now returns I64 (tensor ptr), not F64.
// Update the type tracking so the returned value is treated as a tensor.
```

**Step 4: Add .item() method support**

In the tensor method dispatch in `expr.rs`, add handling for `.item()`:

```rust
if method_name == "item" {
    // Call nsl_tensor_item(tensor_ptr) -> f64
    return self.compile_call_by_name("nsl_tensor_item", &[obj_val], builder);
}
```

This returns F64, which the rest of the codegen can use as a float value.

**Step 5: Fix print dispatch for scalar tensors**

Since `.sum()` now returns a tensor (I64), `print(x.sum())` should dispatch to `nsl_tensor_print` (which already handles 0-d tensors), not `nsl_print_float`. Check how print type dispatch works in the codegen and ensure tensor types route to `nsl_tensor_print`.

The existing `nsl_tensor_print` already handles 0-d tensors:
```rust
if tensor.ndim == 0 {
    if tensor.len > 0 {
        print_float_value(unsafe { *tensor.data });
        println!();
    }
    ...
}
```

So `print(x.sum())` should print `3.0` correctly as long as the codegen dispatches to tensor print.

**Step 6: Verify it compiles**

Run: `cargo build --workspace`
Expected: Compiles successfully

---

## Task 7: Codegen — Grad Block Compilation

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs` (grad block statement compilation)

**Goal:** Compile grad blocks to: start tape → execute body → backward → bind results. No compile-time flag needed — the global tape handles recording automatically.

**Step 1: Implement grad block compilation in stmt.rs**

Replace the no-op `StmtKind::GradBlock(_)` arm in `compile_stmt` (line 175) with:

```rust
StmtKind::GradBlock(grad) => {
    self.compile_grad_block(grad, state, builder)?;
}
```

Then add the `compile_grad_block` method:

```rust
fn compile_grad_block(
    &mut self,
    grad: &nsl_ast::block::GradBlock,
    state: &mut FuncState,
    builder: &mut FunctionBuilder,
) -> Result<(), CodegenError> {
    // 1. Compile the targets expression to get param tensor ptr(s)
    let targets_val = self.compile_expr(&grad.targets, state, builder)?;

    // 2. Build param list (wrap single tensor in a list)
    //    For MVP: assume targets is a single tensor — wrap in a 1-element list.
    //    TODO: support multiple targets (list of tensors) in future.
    let param_list = self.call_runtime("nsl_list_new", &[], builder)?;
    self.call_runtime_void("nsl_list_push", &[param_list, targets_val], builder)?;

    // 3. Start tape recording
    self.call_runtime_void("nsl_tape_start", &[param_list], builder)?;

    // 4. Compile the body — all ops automatically record on the global tape.
    //    The last expression in the body is the loss (a scalar tensor).
    let mut loss_val = None;
    for (i, stmt) in grad.body.stmts.iter().enumerate() {
        if i == grad.body.stmts.len() - 1 {
            // Last statement: if it's an expression, capture its value as loss
            if let StmtKind::Expr(ref expr) = stmt.kind {
                loss_val = Some(self.compile_expr(expr, state, builder)?);
            } else {
                self.compile_stmt(stmt, state, builder)?;
            }
        } else {
            self.compile_stmt(stmt, state, builder)?;
        }
    }

    let loss_tensor = loss_val.ok_or_else(|| {
        CodegenError::new("grad block must end with an expression (the loss)")
    })?;

    // 5. Run backward pass
    let grads_list = self.call_runtime(
        "nsl_tape_backward",
        &[loss_tensor, param_list],
        builder,
    )?;

    // 6. Stop tape (cleans up saved tensor refcounts)
    self.call_runtime_void("nsl_tape_stop", &[], builder)?;

    // 7. Extract scalar loss value via .item()
    let loss_f64 = self.call_runtime(
        "nsl_tensor_item",
        &[loss_tensor],
        builder,
    )?;

    // 8. Get gradient for the single param (index 0 in grads_list)
    //    TODO: support multiple targets — loop over grads_list to unpack
    let zero = builder.ins().iconst(cranelift_codegen::ir::types::I64, 0);
    let grad_tensor = self.call_runtime("nsl_list_get", &[grads_list, zero], builder)?;

    // 9. Bind output variables (if pattern exists)
    if let Some(ref pattern) = grad.outputs {
        match &pattern.kind {
            nsl_ast::pattern::PatternKind::Tuple(pats) if pats.len() == 2 => {
                // Bind loss (f64)
                if let nsl_ast::pattern::PatternKind::Ident(loss_sym) = &pats[0].kind {
                    let var = state.new_variable();
                    builder.declare_var(var, cranelift_codegen::ir::types::F64);
                    builder.def_var(var, loss_f64);
                    state.variables.insert(*loss_sym, (var, cranelift_codegen::ir::types::F64));
                }
                // Bind grads (tensor ptr)
                if let nsl_ast::pattern::PatternKind::Ident(grads_sym) = &pats[1].kind {
                    let var = state.new_variable();
                    builder.declare_var(var, cranelift_codegen::ir::types::I64);
                    builder.def_var(var, grad_tensor);
                    state.variables.insert(*grads_sym, (var, cranelift_codegen::ir::types::I64));
                }
            }
            _ => {
                return Err(CodegenError::new(
                    "grad block output must be `let loss, grads = grad(...):`"
                ));
            }
        }
    }

    Ok(())
}
```

**Important notes for the implementer:**
- `call_runtime` and `call_runtime_void` are helper methods that look up a runtime function by name and emit a Cranelift `call` instruction. Check how existing runtime calls are made (e.g., how `nsl_tensor_add` is called in `expr.rs`) and follow the same pattern. The typical pattern is:
  1. Look up `(func_id, sig)` in `self.functions` or `self.runtime_functions`
  2. Get `func_ref` via `module.declare_func_in_func(func_id, builder.func)`
  3. Call `builder.ins().call(func_ref, &args)`
  4. Extract return value from `builder.inst_results(call)`
- If these helpers don't exist, either create them or inline the pattern.
- `loss_f64` is F64 (Cranelift type), `grad_tensor` is I64 (pointer).

**Step 2: Handle `collect_strings` for grad blocks**

In the `collect_strings_in_stmt` method (search in `compiler.rs`), ensure the `GradBlock` arm walks the body's statements for string interning. If it's currently a no-op, update:

```rust
StmtKind::GradBlock(grad) => {
    for stmt in &grad.body.stmts {
        self.collect_strings_in_stmt(stmt);
    }
}
```

**Step 3: Verify it compiles**

Run: `cargo build --workspace`
Expected: Compiles successfully

---

## Task 8: End-to-End Test — Simple Scalar Gradient

**Files:**
- Create: `examples/m12_grad_basic.nsl`
- Create: `tests/expected/m12_grad_basic.txt`

**Goal:** Verify that `grad` computes correct gradients for elementwise multiply + sum.

**Step 1: Create the test program**

Create `examples/m12_grad_basic.nsl`:

```nsl
# M12 Basic Gradient Test
# d/dw sum(x * w) where x = ones([3]), w = ones([3])
# Forward: x * w = [1, 1, 1], sum = 3.0
# Backward: d/dw = x = [1, 1, 1]

let w = ones([3])
let x = ones([3])

let loss, grads = grad(w):
    let y = x * w
    y.sum()

print(loss)
print(grads)
```

**Step 2: Create expected output**

Create `tests/expected/m12_grad_basic.txt`:
```
3.0
tensor([1.0, 1.0, 1.0])
```

**Step 3: Run it**

Run: `cargo build --workspace && nsl run examples/m12_grad_basic.nsl`
Expected output:
```
3.0
tensor([1.0, 1.0, 1.0])
```

**Step 4: Debug if needed**

Common issues to check:
- Tape not starting (check `nsl_tape_start` is called)
- Tape not recording (check `is_recording()` returns true during body)
- Wrong gradient values (check derivative rules in backward engine)
- `.sum()` return type: must return I64 tensor ptr
- Loss binding: should be F64 after `.item()` extraction
- print dispatch: `print(loss)` should call `nsl_print_float` (loss is f64), `print(grads)` should call `nsl_tensor_print` (grads is tensor)

---

## Task 9: End-to-End Test — Matmul Gradient + Model Integration

**Files:**
- Create: `examples/m12_grad_matmul.nsl`
- Create: `tests/expected/m12_grad_matmul.txt`
- Create: `examples/m12_grad_model.nsl`
- Create: `tests/expected/m12_grad_model.txt`

**Step 1: Create matmul gradient test**

Create `examples/m12_grad_matmul.nsl`:

```nsl
# M12 Matmul Gradient Test
# d/dw sum(x @ w) where x = ones([2, 3]), w = ones([3, 4])
# Forward: x @ w = full([2, 4], 3.0), sum = 24.0
# Backward: d/dw = x^T @ ones([2, 4]) = full([3, 4], 2.0)

let x = ones([2, 3])
let w = ones([3, 4])

let loss, grads = grad(w):
    let y = x @ w
    y.sum()

print(loss)
print(grads)
```

Create `tests/expected/m12_grad_matmul.txt`:
```
24.0
tensor([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
```

**Step 2: Run matmul test**

Run: `nsl run examples/m12_grad_matmul.nsl`
Expected: Output matches expected

**Step 3: Create model gradient test**

Create `examples/m12_grad_model.nsl`:

```nsl
# M12 Model Gradient Test
# Compute gradient of a model's forward pass
# This tests that the global tape transparently records through model.forward()

model Linear():
    weight: Tensor = ones([3, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.weight

let m = Linear()
let x = ones([2, 3])

let loss, grads = grad(m.weight):
    let y = m.forward(x)
    y.sum()

print(loss)
print(grads)
```

Create `tests/expected/m12_grad_model.txt`:
```
24.0
tensor([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
```

**Step 4: Run model test**

Run: `nsl run examples/m12_grad_model.nsl`
Expected: Output matches. This is the critical test — it verifies that `model.forward()` (compiled as a separate function) correctly records its operations on the global tape.

---

## Task 10: @no_grad Decorator Support

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs` or `crates/nsl-codegen/src/func.rs` (function compilation)
- Create: `examples/m12_no_grad.nsl`
- Create: `tests/expected/m12_no_grad.txt`

**Goal:** Functions annotated with `@no_grad` emit `nsl_tape_pause()` at entry and `nsl_tape_resume()` at exit. This prevents gradient tracking through those functions.

**Step 1: Detect @no_grad decorator during function compilation**

Find where function definitions (`StmtKind::FnDef` or `StmtKind::Decorated { stmt: FnDef }`) are compiled into Cranelift functions. When starting to compile the function body, check if the function has an `@no_grad` decorator.

The `@no_grad` decorator appears as `StmtKind::Decorated { decorators, stmt }` where `stmt` is a `FnDef`. The decorator has `name = [Symbol("no_grad")]`.

At the start of the function body, emit `nsl_tape_pause()`. Before each return instruction, emit `nsl_tape_resume()`. For functions with a single implicit return, emit it at the end of the function body.

```rust
// At start of function body (after parameter binding):
if has_no_grad {
    self.call_runtime_void("nsl_tape_pause", &[], builder)?;
}

// Before each return instruction:
if has_no_grad {
    self.call_runtime_void("nsl_tape_resume", &[], builder)?;
}
// ... then emit the actual return
```

**Step 2: Create test program**

Create `examples/m12_no_grad.nsl`:

```nsl
# M12 @no_grad Test
# frozen_transform's operations are NOT recorded on the tape
# So no gradient flows back through it to w

let w = ones([3])
let x = ones([3])

@no_grad
fn frozen_transform(t: Tensor) -> Tensor:
    return t * ones([3])

let loss, grads = grad(w):
    let y = x * w
    let z = frozen_transform(y)
    z.sum()

# z.sum() = 3.0 (same computation)
# But gradient of z w.r.t. w is zero because the tape has no record
# of how z was produced from y (frozen_transform was @no_grad)
print(loss)
print(grads)
```

Create `tests/expected/m12_no_grad.txt`:
```
3.0
tensor([0.0, 0.0, 0.0])
```

**Step 3: Run test**

Run: `nsl run examples/m12_no_grad.nsl`
Expected: `3.0` then `tensor([0.0, 0.0, 0.0])`

---

## Task 11: Update Existing Examples for sum/mean Change + Regression Verification

**Files:**
- Modify: Any existing examples/tests that use `.sum()` or `.mean()` expecting f64
- Modify: Codegen in `expr.rs` if type tracking for sum/mean results needs fixing

**Goal:** Ensure the sum/mean return type change doesn't break existing programs.

**Step 1: Identify affected examples**

Search all `.nsl` example files for `.sum()` and `.mean()` usage. Check if the result is used as a float (e.g., `print(tensor.sum())`) or as a tensor.

**Step 2: Fix affected call sites**

The key issue: if `print(x.sum())` was previously compiled as `nsl_print_float(nsl_tensor_sum(x))`, it now needs to be `nsl_tensor_print(nsl_tensor_sum(x))` since sum returns a tensor. The existing `nsl_tensor_print` already handles 0-d tensors correctly (prints just the scalar value like `3.0`).

Ensure the codegen's type tracking correctly identifies `.sum()` results as tensor type (I64) so print dispatch routes to `nsl_tensor_print`.

**Step 3: Run ALL existing examples**

```bash
nsl run examples/hello.nsl
nsl run examples/features.nsl
nsl run examples/m5_features.nsl
nsl run examples/m6_features.nsl
nsl run examples/m8_features.nsl
nsl run examples/m9_tensors.nsl
nsl run examples/m11_model_basic.nsl
nsl run examples/m11_model_tensor.nsl
nsl run examples/m11_forward_dispatch.nsl
nsl run examples/m11_model_compose.nsl
```

Expected: All produce correct output matching `tests/expected/` files.

**Step 4: Run M10 shape checking**

Run: `nsl check examples/m10_shape_errors.nsl`
Expected: Still catches all 4 shape errors.

**Step 5: Run all M12 tests**

```bash
nsl run examples/m12_grad_basic.nsl
nsl run examples/m12_grad_matmul.nsl
nsl run examples/m12_grad_model.nsl
nsl run examples/m12_no_grad.nsl
```

Expected: All produce correct output.

**Step 6: Run unit tests**

Run: `cargo test --workspace`
Expected: All tests pass.

---

## Verification Checklist

- [ ] `cargo test --workspace` — all tests pass
- [ ] `nsl run examples/m12_grad_basic.nsl` — prints `3.0` and `tensor([1.0, 1.0, 1.0])`
- [ ] `nsl run examples/m12_grad_matmul.nsl` — prints `24.0` and 3x4 tensor of 2.0s
- [ ] `nsl run examples/m12_grad_model.nsl` — prints `24.0` and 3x4 tensor of 2.0s (tests global tape through model calls)
- [ ] `nsl run examples/m12_no_grad.nsl` — prints `3.0` and tensor of 0.0s
- [ ] All M1-M11 examples produce correct output (no regressions from sum/mean change)
- [ ] `nsl check examples/m10_shape_errors.nsl` — still catches shape errors

---

## File Summary

| File | Action | Purpose |
|------|--------|---------|
| `crates/nsl-runtime/src/autodiff.rs` | Create | Tape, backward engine, start/stop/pause/resume |
| `crates/nsl-runtime/src/lib.rs` | Modify | Add `pub mod autodiff;` |
| `crates/nsl-runtime/src/tensor.rs` | Modify | sum/mean return scalar tensors, add .item(), tape recording in all ops |
| `crates/nsl-ast/src/block.rs` | Modify | Add `outputs` field to GradBlock |
| `crates/nsl-parser/src/stmt.rs` | Modify | Parse `let pat = grad(targets):` |
| `crates/nsl-parser/src/block.rs` | Modify | Update parse_grad_block_stmt |
| `crates/nsl-semantic/src/checker.rs` | Modify | Grad block validation + output bindings |
| `crates/nsl-codegen/src/builtins.rs` | Modify | Declare AD runtime functions, update sum/mean return types |
| `crates/nsl-codegen/src/stmt.rs` | Modify | Grad block compilation |
| `crates/nsl-codegen/src/expr.rs` | Modify | Update .sum()/.mean() return type handling, add .item() method |
| `examples/m12_grad_basic.nsl` | Create | Simple gradient test |
| `examples/m12_grad_matmul.nsl` | Create | Matmul gradient test |
| `examples/m12_grad_model.nsl` | Create | Model gradient test (global tape through function calls) |
| `examples/m12_no_grad.nsl` | Create | @no_grad test |
| `tests/expected/m12_grad_basic.txt` | Create | Expected output |
| `tests/expected/m12_grad_matmul.txt` | Create | Expected output |
| `tests/expected/m12_grad_model.txt` | Create | Expected output |
| `tests/expected/m12_no_grad.txt` | Create | Expected output |
