# M18a: Transformer Foundations Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tensor ops, model composition, and stdlib layers to define and run a multi-layer transformer in pure NSL.

**Architecture:** 5 new Rust runtime tensor ops (unsqueeze, expand, stack, select, causal_mask) with autodiff, nested model fields via opaque pointer pattern, fixed-size model arrays with literal-only sizing, and stdlib transformer block built on these primitives.

**Tech Stack:** Rust (nsl-runtime), Cranelift IR (nsl-codegen), NSL stdlib

**Spec:** `docs/superpowers/specs/2026-03-11-m18a-transformer-foundations-design.md`

---

## Chunk 1: Runtime Tensor Operations

### Task 1: `nsl_tensor_unsqueeze`

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`

- [ ] **Step 1: Write test program**

Create `tests/m18_unsqueeze_test.nsl`:
```nsl
# Test unsqueeze
let a = full([3, 4], 1.0)

# unsqueeze at dim 0: [3, 4] -> [1, 3, 4]
let b = unsqueeze(a, 0)
print(b)

# unsqueeze at dim 1: [3, 4] -> [3, 1, 4]
let c = unsqueeze(a, 1)
print(c)

# unsqueeze at dim -1: [3, 4] -> [3, 4, 1]
let d = unsqueeze(a, -1)
print(d)

print("unsqueeze: PASS")
```

- [ ] **Step 2: Implement `nsl_tensor_unsqueeze` in tensor.rs**

Add after existing `nsl_tensor_transpose` function. Follow the same pattern as `nsl_tensor_reshape`:

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_unsqueeze(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = unsafe { &*(tensor_ptr as *const NslTensor) };
    let ndim = tensor.ndim as usize;

    // Normalize negative dim
    let dim = if dim < 0 { (ndim as i64 + 1 + dim) as usize } else { dim as usize };
    assert!(dim <= ndim, "unsqueeze dim {} out of range for {}-D tensor", dim, ndim);

    // Build new shape: insert 1 at dim
    let old_shape = unsafe { std::slice::from_raw_parts(tensor.shape, ndim) };
    let new_ndim = ndim + 1;
    let new_shape_data = checked_alloc(new_ndim * 8) as *mut i64;
    let new_strides_data = checked_alloc(new_ndim * 8) as *mut i64;

    let mut new_shape = Vec::with_capacity(new_ndim);
    for i in 0..new_ndim {
        if i < dim {
            new_shape.push(old_shape[i]);
        } else if i == dim {
            new_shape.push(1);
        } else {
            new_shape.push(old_shape[i - 1]);
        }
    }

    // Compute strides for new shape
    let mut stride = 1i64;
    for i in (0..new_ndim).rev() {
        unsafe { *new_strides_data.add(i) = stride; }
        unsafe { *new_shape_data.add(i) = new_shape[i]; }
        stride *= new_shape[i];
    }

    // Copy data (deep copy to maintain ownership invariant)
    let total_len = tensor.len as usize;
    let result = if tensor.dtype == 1 {
        let data_size = total_len * std::mem::size_of::<f32>();
        let new_data = checked_alloc(data_size) as *mut f32;
        unsafe { std::ptr::copy_nonoverlapping(tensor.data as *const f32, new_data, total_len); }
        alloc_tensor(new_data as *mut c_void, new_shape_data, new_strides_data, new_ndim as i64, total_len as i64, tensor.device, tensor.dtype)
    } else {
        let data_size = total_len * std::mem::size_of::<f64>();
        let new_data = checked_alloc(data_size) as *mut f64;
        unsafe { std::ptr::copy_nonoverlapping(tensor.data as *const f64, new_data, total_len); }
        alloc_tensor(new_data as *mut c_void, new_shape_data, new_strides_data, new_ndim as i64, total_len as i64, tensor.device, tensor.dtype)
    };

    // Record on tape (no saved refs needed — backward is just reshape)
    record_tape_op_unsqueeze(tensor_ptr, result, dim as i64);

    result
}
```

Note: `alloc_tensor` is a helper that boxes an NslTensor and returns i64 pointer. If this helper doesn't exist, build the tensor manually following the pattern in `nsl_tensor_reshape`. `record_tape_op_unsqueeze` is added in Task 6.

- [ ] **Step 3: Register in builtins.rs**

Add to RUNTIME_FUNCTIONS array in `crates/nsl-codegen/src/builtins.rs`:
```rust
("nsl_tensor_unsqueeze", &[types::I64, types::I64], Some(types::I64)),
```

- [ ] **Step 4: Add codegen dispatch**

In `crates/nsl-codegen/src/expr.rs`, in the tensor method call handler (same area as `reshape`, `transpose`, `clone`, `slice`), add:
```rust
"unsqueeze" => {
    if args.len() != 1 {
        return Err(CodegenError::new("unsqueeze() takes exactly 1 argument (dim)"));
    }
    let dim_val = self.compile_expr(builder, state, &args[0].value)?;
    self.compile_call_by_name(builder, "nsl_tensor_unsqueeze", &[obj_val, dim_val])
}
```

Also add `unsqueeze` as a free function in the function call handler (same area as `zeros`, `ones`, `reshape`):
```rust
"unsqueeze" => {
    // unsqueeze(tensor, dim)
    let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
    let dim_val = self.compile_expr(builder, state, &args[1].value)?;
    self.compile_call_by_name(builder, "nsl_tensor_unsqueeze", &[tensor_val, dim_val])
}
```

- [ ] **Step 5: Run test to verify**

Run: `cargo run -p nsl-cli -- run tests/m18_unsqueeze_test.nsl`
Expected: Prints tensors with correct shapes, then "unsqueeze: PASS"

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr.rs tests/m18_unsqueeze_test.nsl
git commit -m "feat(m18a): add nsl_tensor_unsqueeze runtime op + codegen"
```

---

### Task 2: `nsl_tensor_select`

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

- [ ] **Step 1: Write test program**

Create `tests/m18_select_test.nsl`:
```nsl
let a = full([3, 4], 1.0)

# Select index 0 along dim 0: [3, 4] -> [4]
let b = a.select(0, 0)
print(b)

# Select index 1 along dim 1: [3, 4] -> [3]
let c = a.select(1, 2)
print(c)

# 3D test
let d = full([2, 3, 4], 2.0)
let e = d.select(0, 1)
print(e)

print("select: PASS")
```

- [ ] **Step 2: Implement `nsl_tensor_select` in tensor.rs**

Follow the `nsl_tensor_slice` pattern. `select(dim, index)` extracts a hyperplane at `index` along `dim`, removing that dimension.

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_select(tensor_ptr: i64, dim: i64, index: i64) -> i64 {
    let tensor = unsafe { &*(tensor_ptr as *const NslTensor) };
    let ndim = tensor.ndim as usize;
    let shape = unsafe { std::slice::from_raw_parts(tensor.shape, ndim) };

    // Normalize negative dim and index
    let dim = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    let index = if index < 0 { (shape[dim] + index) as usize } else { index as usize };
    assert!(dim < ndim, "select dim out of range");
    assert!((index as i64) < shape[dim], "select index out of range");

    // Output shape: remove the selected dimension
    let new_ndim = ndim - 1;
    let mut out_shape = Vec::with_capacity(new_ndim);
    for d in 0..ndim {
        if d != dim { out_shape.push(shape[d]); }
    }
    let out_len: i64 = out_shape.iter().product();

    // Allocate output
    let new_shape_data = checked_alloc(new_ndim * 8) as *mut i64;
    let new_strides_data = checked_alloc(new_ndim * 8) as *mut i64;
    for (i, &s) in out_shape.iter().enumerate() {
        unsafe { *new_shape_data.add(i) = s; }
    }
    let mut stride = 1i64;
    for i in (0..new_ndim).rev() {
        unsafe { *new_strides_data.add(i) = stride; }
        stride *= out_shape[i];
    }

    // Copy data: iterate over all output indices, map to input indices
    // Input index = output index with `index` inserted at position `dim`
    if tensor.dtype == 1 {
        let src = tensor.data as *const f32;
        let data_size = (out_len as usize) * std::mem::size_of::<f32>();
        let dst = checked_alloc(data_size) as *mut f32;
        // Use flat iteration: for each output element, compute source offset
        copy_select_data_f32(src, dst, shape, &out_shape, dim, index, ndim);
        let result = alloc_tensor(dst as *mut c_void, new_shape_data, new_strides_data, new_ndim as i64, out_len, tensor.device, tensor.dtype);
        result
    } else {
        let src = tensor.data as *const f64;
        let data_size = (out_len as usize) * std::mem::size_of::<f64>();
        let dst = checked_alloc(data_size) as *mut f64;
        copy_select_data_f64(src, dst, shape, &out_shape, dim, index, ndim);
        let result = alloc_tensor(dst as *mut c_void, new_shape_data, new_strides_data, new_ndim as i64, out_len, tensor.device, tensor.dtype);
        result
    }
    // Note: select does NOT record a TapeOp — it's only used internally for stack backward
}
```

The `copy_select_data_f64/f32` helpers iterate output indices and compute corresponding input offsets. Use the same multi-dimensional indexing pattern as `nsl_tensor_slice`.

- [ ] **Step 3: Register in builtins.rs and add codegen dispatch**

builtins.rs:
```rust
("nsl_tensor_select", &[types::I64, types::I64, types::I64], Some(types::I64)),
```

expr.rs tensor method call:
```rust
"select" => {
    if args.len() != 2 {
        return Err(CodegenError::new("select() takes 2 arguments (dim, index)"));
    }
    let dim_val = self.compile_expr(builder, state, &args[0].value)?;
    let idx_val = self.compile_expr(builder, state, &args[1].value)?;
    self.compile_call_by_name(builder, "nsl_tensor_select", &[obj_val, dim_val, idx_val])
}
```

- [ ] **Step 4: Run test**

Run: `cargo run -p nsl-cli -- run tests/m18_select_test.nsl`

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr.rs tests/m18_select_test.nsl
git commit -m "feat(m18a): add nsl_tensor_select runtime op + codegen"
```

---

### Task 3: `nsl_tensor_stack`

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

- [ ] **Step 1: Write test program**

Create `tests/m18_stack_test.nsl`:
```nsl
let a = full([4], 1.0)
let b = full([4], 2.0)
let c = full([4], 3.0)

# stack along dim 0: three [4] tensors -> [3, 4]
let s = stack([a, b, c], 0)
print(s)

# 2D stack
let x = full([2, 3], 1.0)
let y = full([2, 3], 2.0)
let s2 = stack([x, y], 0)
print(s2)

print("stack: PASS")
```

- [ ] **Step 2: Implement `nsl_tensor_stack`**

Signature: `nsl_tensor_stack(list_ptr: i64, dim: i64) -> i64` where `list_ptr` is an NslList of tensor pointers.

Follow the `nsl_tensor_cat` pattern but insert a new dimension. Implementation:
1. Extract tensor pointers from the NslList
2. Validate all shapes are identical
3. Compute output shape: insert `count` at position `dim`
4. Allocate output tensor
5. For each input tensor, copy data into the correct region of output
6. Record TapeOp::Stack for autodiff

- [ ] **Step 3: Register + codegen**

builtins.rs:
```rust
("nsl_tensor_stack", &[types::I64, types::I64], Some(types::I64)),
```

expr.rs free function call handler (where `cat` is handled):
```rust
"stack" => {
    // stack(list_of_tensors, dim)
    let list_val = self.compile_expr(builder, state, &args[0].value)?;
    let dim_val = self.compile_expr(builder, state, &args[1].value)?;
    self.compile_call_by_name(builder, "nsl_tensor_stack", &[list_val, dim_val])
}
```

- [ ] **Step 4: Run test**

Run: `cargo run -p nsl-cli -- run tests/m18_stack_test.nsl`

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr.rs tests/m18_stack_test.nsl
git commit -m "feat(m18a): add nsl_tensor_stack runtime op + codegen"
```

---

### Task 4: `nsl_tensor_expand`

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

- [ ] **Step 1: Write test program**

Create `tests/m18_expand_test.nsl`:
```nsl
# expand [1, 4] -> [3, 4]
let a = full([1, 4], 2.0)
let b = a.expand([3, 4])
print(b)

# expand [4] -> [3, 4] (prepend dimension)
let c = full([4], 1.0)
let d = c.expand([3, 4])
print(d)

print("expand: PASS")
```

- [ ] **Step 2: Implement `nsl_tensor_expand`**

Signature: `nsl_tensor_expand(tensor_ptr: i64, shape_list: i64) -> i64` where `shape_list` is an NslList of ints (same pattern as reshape).

Implementation:
1. Extract target shape from NslList
2. Right-align source shape with target shape (pad left with 1s if needed)
3. For each dimension: if source=1 and target>1, replicate. If source==target, copy. Otherwise error.
4. Allocate output with target shape
5. Copy data with replication
6. Record TapeOp::Expand with original_shape for backward

- [ ] **Step 3: Register + codegen**

builtins.rs:
```rust
("nsl_tensor_expand", &[types::I64, types::I64], Some(types::I64)),
```

expr.rs tensor method:
```rust
"expand" => {
    if args.len() != 1 {
        return Err(CodegenError::new("expand() takes 1 argument (target shape list)"));
    }
    let shape_val = self.compile_expr(builder, state, &args[0].value)?;
    self.compile_call_by_name(builder, "nsl_tensor_expand", &[obj_val, shape_val])
}
```

- [ ] **Step 4: Run test**

Run: `cargo run -p nsl-cli -- run tests/m18_expand_test.nsl`

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr.rs tests/m18_expand_test.nsl
git commit -m "feat(m18a): add nsl_tensor_expand runtime op + codegen"
```

---

### Task 5: `nsl_tensor_causal_mask`

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

- [ ] **Step 1: Write test program**

Create `tests/m18_causal_mask_test.nsl`:
```nsl
# 4x4 causal mask
let mask = causal_mask(4)
print(mask)
# Expected: lower triangle = 0.0, upper triangle = -1000000000.0

print("causal_mask: PASS")
```

- [ ] **Step 2: Implement `nsl_tensor_causal_mask`**

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_causal_mask(seq_len: i64) -> i64 {
    let n = seq_len as usize;
    let total = n * n;
    let data_size = total * std::mem::size_of::<f64>();
    let data = checked_alloc(data_size) as *mut f64;

    for i in 0..n {
        for j in 0..n {
            unsafe {
                *data.add(i * n + j) = if j <= i { 0.0 } else { -1e9 };
            }
        }
    }

    let shape_data = checked_alloc(2 * 8) as *mut i64;
    let strides_data = checked_alloc(2 * 8) as *mut i64;
    unsafe {
        *shape_data = seq_len;
        *shape_data.add(1) = seq_len;
        *strides_data = seq_len;
        *strides_data.add(1) = 1;
    }

    alloc_tensor(data as *mut c_void, shape_data, strides_data, 2, total as i64, 0, 0)
    // device=0 (CPU), dtype=0 (f64), NOT recorded on tape
}
```

- [ ] **Step 3: Register + codegen**

builtins.rs:
```rust
("nsl_tensor_causal_mask", &[types::I64], Some(types::I64)),
```

expr.rs free function:
```rust
"causal_mask" => {
    let seq_val = self.compile_expr(builder, state, &args[0].value)?;
    self.compile_call_by_name(builder, "nsl_tensor_causal_mask", &[seq_val])
}
```

- [ ] **Step 4: Run test**

Run: `cargo run -p nsl-cli -- run tests/m18_causal_mask_test.nsl`

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr.rs tests/m18_causal_mask_test.nsl
git commit -m "feat(m18a): add nsl_tensor_causal_mask runtime op + codegen"
```

---

### Task 6: Run existing tests (regression check)

- [ ] **Step 1: Run cargo test**

Run: `cargo test --workspace`
Expected: All existing tests pass

- [ ] **Step 2: Run integration tests**

Run: `cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl`
Run: `cargo run -p nsl-cli -- run examples/m15_tiny_lm.nsl`
Run: `cargo run -p nsl-cli -- run examples/m16_quantize.nsl`
Expected: All produce correct output matching `tests/expected/`

- [ ] **Step 3: Commit if any fixes needed**

---

## Chunk 2: Autodiff Backward Passes

### Task 7: TapeOp::Unsqueeze backward

**Files:**
- Modify: `crates/nsl-runtime/src/autodiff.rs`

- [ ] **Step 1: Add TapeOp::Unsqueeze variant**

In the `TapeOp` enum (autodiff.rs):
```rust
Unsqueeze {
    result: i64,   // TensorId of output
    input: i64,    // TensorId of input
    dim: i64,      // which dimension was inserted
    input_shape: Vec<i64>,  // original shape for reshape backward
},
```

- [ ] **Step 2: Add recording function**

```rust
fn record_tape_op_unsqueeze(input_ptr: i64, result_ptr: i64, dim: i64) {
    let input = unsafe { &*(input_ptr as *const NslTensor) };
    let input_shape = unsafe { std::slice::from_raw_parts(input.shape, input.ndim as usize) }.to_vec();
    // Bump refcount on input (saved for backward shape info — not data)
    // Actually we save shape as Vec<i64>, no need to save tensor ref
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        if tape.recording && tape.pause_depth == 0 {
            tape.ops.push(TapeOp::Unsqueeze {
                result: result_ptr,
                input: input_ptr,
                dim,
                input_shape,
            });
        }
    });
}
```

- [ ] **Step 3: Add backward pass**

In the backward match block (inside `nsl_tape_backward`):
```rust
TapeOp::Unsqueeze { result, input, dim: _, input_shape } => {
    if let Some(&g) = grad_map.get(result) {
        // Backward is reshape to original shape
        let grad_input = nsl_tensor_reshape_raw(g, input_shape);
        accumulate_grad(&mut grad_map, *input, grad_input);
    }
}
```

`nsl_tensor_reshape_raw` takes a tensor pointer and a `&[i64]` shape slice (internal helper, not FFI). If this doesn't exist, implement it by extracting the reshape logic from `nsl_tensor_reshape`.

- [ ] **Step 4: Add to `release_tape_op_refs`**

In the `release_tape_op_refs` function, add:
```rust
TapeOp::Unsqueeze { .. } => {
    // No saved tensor refs (shape saved as Vec<i64>)
}
```

- [ ] **Step 5: Call recorder from unsqueeze**

Update `nsl_tensor_unsqueeze` (Task 1) to call `record_tape_op_unsqueeze(tensor_ptr, result, dim)`.

- [ ] **Step 6: Write autodiff test**

Create `tests/m18_unsqueeze_grad_test.nsl`:
```nsl
let x = full([3, 4], 2.0)

let loss, grads = grad(x):
    let y = unsqueeze(x, 0)    # [1, 3, 4]
    let z = sum(y)
    z

print(grads)
print("unsqueeze grad: PASS")
```

Run: `cargo run -p nsl-cli -- run tests/m18_unsqueeze_grad_test.nsl`

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-runtime/src/autodiff.rs crates/nsl-runtime/src/tensor.rs tests/m18_unsqueeze_grad_test.nsl
git commit -m "feat(m18a): add TapeOp::Unsqueeze backward pass"
```

---

### Task 8: TapeOp::Expand backward

**Files:**
- Modify: `crates/nsl-runtime/src/autodiff.rs`
- Modify: `crates/nsl-runtime/src/tensor.rs`

- [ ] **Step 1: Add TapeOp::Expand variant**

```rust
Expand {
    result: i64,
    input: i64,
    original_shape: Vec<i64>,  // pre-expand shape for reduce-sum backward
},
```

- [ ] **Step 2: Add recording in `nsl_tensor_expand`**

Record original shape before expand. No tensor refs needed.

- [ ] **Step 3: Add backward pass**

```rust
TapeOp::Expand { result, input, original_shape } => {
    if let Some(&g) = grad_map.get(result) {
        // Reduce-sum along each broadcast dimension (reverse order)
        let mut grad_input = g;
        let grad_tensor = unsafe { &*(g as *const NslTensor) };
        let grad_shape = unsafe { std::slice::from_raw_parts(grad_tensor.shape, grad_tensor.ndim as usize) };

        // Right-align: if original had fewer dims, pad left with 1s
        let offset = grad_shape.len() - original_shape.len();
        for d in (0..grad_shape.len()).rev() {
            let orig_d = if d >= offset { original_shape[d - offset] } else { 1 };
            if orig_d == 1 && grad_shape[d] > 1 {
                grad_input = nsl_tensor_sum_dim(grad_input, d as i64, 1); // keepdim=true
            }
        }
        // Reshape to original shape
        grad_input = nsl_tensor_reshape_raw(grad_input, &original_shape);
        accumulate_grad(&mut grad_map, *input, grad_input);
    }
}
```

- [ ] **Step 4: Write test**

Create `tests/m18_expand_grad_test.nsl`:
```nsl
let x = full([1, 4], 3.0)

let loss, grads = grad(x):
    let y = x.expand([3, 4])
    let z = sum(y)
    z

# Gradient should be [3, 3, 3, 3] (each element contributes to 3 outputs)
print(grads)
print("expand grad: PASS")
```

Run: `cargo run -p nsl-cli -- run tests/m18_expand_grad_test.nsl`

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/autodiff.rs crates/nsl-runtime/src/tensor.rs tests/m18_expand_grad_test.nsl
git commit -m "feat(m18a): add TapeOp::Expand backward pass"
```

---

### Task 9: TapeOp::Stack backward

**Files:**
- Modify: `crates/nsl-runtime/src/autodiff.rs`
- Modify: `crates/nsl-runtime/src/tensor.rs`

- [ ] **Step 1: Add TapeOp::Stack variant**

```rust
Stack {
    result: i64,
    inputs: Vec<i64>,   // TensorIds of all stacked inputs (refcounted)
    dim: i64,
},
```

- [ ] **Step 2: Add recording in `nsl_tensor_stack`**

Save all input tensor pointers (bump refcount on each for tape safety). Record `dim`.

- [ ] **Step 3: Add backward pass**

```rust
TapeOp::Stack { result, inputs, dim } => {
    if let Some(&g) = grad_map.get(result) {
        for (i, input_id) in inputs.iter().enumerate() {
            let grad_piece = nsl_tensor_select(g, *dim, i as i64);
            accumulate_grad(&mut grad_map, *input_id, grad_piece);
        }
    }
}
```

- [ ] **Step 4: Add to `release_tape_op_refs`**

```rust
TapeOp::Stack { inputs, .. } => {
    for &input in inputs {
        tensor_free(input);
    }
}
```

- [ ] **Step 5: Write test**

Create `tests/m18_stack_grad_test.nsl`:
```nsl
let a = full([4], 1.0)
let b = full([4], 2.0)

let loss, grads = grad(a, b):
    let s = stack([a, b], 0)  # [2, 4]
    let z = sum(s)
    z

print(grads)
print("stack grad: PASS")
```

Run: `cargo run -p nsl-cli -- run tests/m18_stack_grad_test.nsl`

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/autodiff.rs crates/nsl-runtime/src/tensor.rs tests/m18_stack_grad_test.nsl
git commit -m "feat(m18a): add TapeOp::Stack backward pass"
```

---

### Task 10: Regression check after autodiff changes

- [ ] **Step 1: Run all tests**

Run: `cargo test --workspace`
Run: `cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl`
Run: `cargo run -p nsl-cli -- test tests/m15_test.nsl`
Expected: All pass

- [ ] **Step 2: Commit if fixes needed**

---

## Chunk 3: Nested Model Fields

### Task 11: Semantic checker — allow Model-typed fields

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Understand current model field type resolution**

Read `check_model_def` in checker.rs. Currently, field type annotations are resolved via `self.resolve_type(type_ann)`. If the type name matches a defined model name, it may resolve to `Type::Unknown` or error.

- [ ] **Step 2: Add model name resolution in field type checking**

When resolving a field's type annotation in `check_model_def`, if the type name matches a previously declared model, resolve it to `Type::Model { name, fields, methods }`. The model definitions are available in `self.models` or equivalent registry.

Key: The semantic checker must have already processed the referenced model's definition before the referencing model. If models are checked in declaration order, this works as long as models are defined before use (no forward references).

- [ ] **Step 3: Write test program**

Create `tests/m18_nested_model_test.nsl`:
```nsl
model Inner(dim: int):
    w: Tensor = randn([dim, dim])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

model Outer(dim: int):
    inner: Inner = Inner(dim)
    b: Tensor = zeros([dim])

    fn forward(self, x: Tensor) -> Tensor:
        let h = self.inner.forward(x)
        return bias_add(h, self.b)

let m = Outer(4)
let x = ones([3, 4])
let y = m.forward(x)
print(y)
print("nested model: PASS")
```

- [ ] **Step 4: Run test (will likely fail at codegen stage — chained access)**

Run: `cargo run -p nsl-cli -- run tests/m18_nested_model_test.nsl`
Expected: May fail on `self.inner.forward(x)` — this is fixed in next task.

- [ ] **Step 5: Commit semantic changes**

```bash
git add crates/nsl-semantic/src/checker.rs tests/m18_nested_model_test.nsl
git commit -m "feat(m18a): allow Model-typed fields in semantic checker"
```

---

### Task 12: Codegen — chained model method calls

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

This is the most complex task in M18a. The goal: make `self.inner.forward(x)` work.

- [ ] **Step 1: Add field-to-model-type lookup table**

In `compiler.rs`, add a field to the Compiler struct:
```rust
/// model_name -> field_name -> field's model type name
pub model_field_types: HashMap<String, HashMap<String, String>>,
```

- [ ] **Step 2: Populate the table in `collect_models()`**

After building the StructLayout, iterate fields again. For each field whose type annotation is a `Named(sym)` that matches a known model name, record it:

```rust
// Inside collect_models(), after building StructLayout
let mut field_types = HashMap::new();
for member in &md.members {
    if let ModelMember::LayerDecl { name: field_sym, type_ann, .. } = member {
        let field_name = self.resolve_sym(*field_sym).to_string();
        if let TypeExprKind::Named(type_sym) = &type_ann.kind {
            let type_name = self.resolve_sym(*type_sym).to_string();
            // Check if type_name is a known model
            if self.struct_layouts.contains_key(&type_name) || /* check model registry */ {
                field_types.insert(field_name, type_name);
            }
        }
    }
}
self.model_field_types.insert(name.clone(), field_types);
```

Note: Models must be processed in dependency order. If `Inner` is defined before `Outer`, `collect_models` processes them in order, so `struct_layouts` already has `Inner` when processing `Outer`.

- [ ] **Step 3: Handle chained member access in expr.rs**

In `compile_call` (expr.rs), when the callee is `MemberAccess(MemberAccess(self, field_name), method_name)`:

The current code handles one level of MemberAccess for model method calls. For chained access, we need to detect the pattern and look up the intermediate model type.

In the match arm that handles `ExprKind::MemberAccess { object, member }` inside `compile_call`:

```rust
// Current: handles model_var.method(args)
// Need to also handle: model_var.field.method(args) where field is a model

// When object is itself a MemberAccess:
if let ExprKind::MemberAccess { object: inner_obj, member: inner_member } = &object.kind {
    let inner_member_name = self.resolve_sym(*inner_member).to_string();

    // Check if inner_obj is a model and inner_member is a model-typed field
    let inner_type = self.node_type(inner_obj.id);
    if let Type::Model { name: parent_model, .. } = inner_type {
        let parent_name = self.resolve_sym(*parent_model).to_string();
        if let Some(field_model_name) = self.model_field_types
            .get(&parent_name)
            .and_then(|fields| fields.get(&inner_member_name))
        {
            // This is model_var.sub_model_field.method()
            // Compile inner_obj.inner_member to get sub-model pointer
            let sub_model_ptr = self.compile_member_access_inner(builder, state, inner_obj, *inner_member)?;
            // Now call the method on the sub-model
            return self.compile_model_method_call(
                builder, state,
                /* pass sub_model_ptr directly instead of recompiling object */
                ...
            );
        }
    }
}
```

The exact implementation depends on how `compile_model_method_call` accepts its self pointer. Currently it takes an `&Expr` for the object and compiles it. For chained access, we need a variant that takes a pre-compiled `Value` instead:

```rust
fn compile_model_method_call_with_ptr(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    self_val: Value,
    model_name: &str,
    method_name: &str,
    args: &[nsl_ast::expr::Arg],
) -> Result<Value, CodegenError> {
    let mangled = self.model_methods.get(model_name)
        .and_then(|m| m.get(method_name))
        .ok_or_else(|| CodegenError::new(format!("no method {method_name} on {model_name}")))?
        .clone();
    let mut arg_vals = vec![self_val];
    for arg in args {
        arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
    }
    self.compile_call_by_name(builder, &mangled, &arg_vals)
}
```

- [ ] **Step 4: Run nested model test**

Run: `cargo run -p nsl-cli -- run tests/m18_nested_model_test.nsl`
Expected: Prints tensor result, then "nested model: PASS"

- [ ] **Step 5: Run regression tests**

Run: `cargo test --workspace`
Run: `cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/compiler.rs crates/nsl-codegen/src/expr.rs
git commit -m "feat(m18a): support chained model method calls (self.sub.forward)"
```

---

## Chunk 4: Fixed-Size Model Arrays

### Task 13: Parser — `[Type; N]` type expression

**Files:**
- Modify: `crates/nsl-ast/src/types.rs`
- Modify: `crates/nsl-parser/src/types.rs` (or wherever `parse_type` is defined)

- [ ] **Step 1: Add FixedArray variant to TypeExprKind**

In `crates/nsl-ast/src/types.rs`:
```rust
/// Fixed-size array type: [TransformerBlock; 12]
FixedArray {
    element_type: Box<TypeExpr>,
    size: i64,
},
```

- [ ] **Step 2: Add parsing in `parse_type`**

Find where `parse_type` is defined (likely `crates/nsl-parser/src/types.rs`). When the current token is `[`:

```rust
TokenKind::LBracket => {
    p.advance(); // consume [
    let elem_type = parse_type(p);
    p.expect(&TokenKind::Semicolon);
    let size = match p.current_kind() {
        TokenKind::IntLit(n) => { let n = *n; p.advance(); n }
        _ => panic!("expected integer literal in fixed array type"),
    };
    p.expect(&TokenKind::RBracket);
    TypeExpr {
        kind: TypeExprKind::FixedArray {
            element_type: Box::new(elem_type),
            size,
        },
        span: start.merge(p.prev_span()),
    }
}
```

Note: Check if `TokenKind::Semicolon` exists in the lexer. If not, the parser may need to use a different delimiter. Check the lexer for `;` support. If `;` is not lexed, use a different syntax like `[Type * N]` or add `;` to the lexer.

- [ ] **Step 3: Test parsing**

Run: `cargo run -p nsl-cli -- check tests/m18_nested_model_test.nsl` (should still work)
Create a minimal test that uses the new syntax and run `check` to verify parsing.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-ast/src/types.rs crates/nsl-parser/src/types.rs
git commit -m "feat(m18a): parse [Type; N] fixed-size array type expression"
```

---

### Task 14: Semantic checker — FixedModelArray type

**Files:**
- Modify: `crates/nsl-semantic/src/types.rs`
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Add FixedModelArray variant to Type enum**

In `crates/nsl-semantic/src/types.rs`:
```rust
/// Fixed-size array of models: [TransformerBlock; 12]
FixedModelArray {
    element_model: Symbol,
    size: i64,
},
```

- [ ] **Step 2: Handle FixedArray in type resolution**

In checker.rs, when resolving a `TypeExprKind::FixedArray` in the model field context:
```rust
TypeExprKind::FixedArray { element_type, size } => {
    let elem_type = self.resolve_type(element_type);
    if let Type::Model { name, .. } = &elem_type {
        Type::FixedModelArray { element_model: *name, size: *size }
    } else {
        self.error("fixed array element must be a model type");
        Type::Error
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-semantic/src/types.rs crates/nsl-semantic/src/checker.rs
git commit -m "feat(m18a): add FixedModelArray type to semantic checker"
```

---

### Task 15: Codegen — FixedArray layout + constructor loop

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 1: Handle FixedArray in `collect_models()`**

When a field's type_ann is `TypeExprKind::FixedArray { element_type, size }`:

```rust
TypeExprKind::FixedArray { element_type, size } => {
    // Each element is stored as a pointer (I64, 8 bytes)
    let total_array_bytes = (*size as usize) * 8;
    let cl_type = cl_types::I64; // for offset alignment calculation
    let align = 8usize;
    offset = (offset + align - 1) & !(align - 1);

    // Store a single StructField with the total array size
    // The field name still maps to the offset of the first element
    fields.push(StructField {
        name: field_name.clone(),
        cl_type,
        offset,
    });
    offset += total_array_bytes;

    // Also record in model_field_types with array info
    if let TypeExprKind::Named(elem_sym) = &element_type.kind {
        let elem_name = self.resolve_sym(*elem_sym).to_string();
        // Store as special marker, e.g., "[ModelName;N]"
        field_types.insert(field_name, format!("[{};{}]", elem_name, size));
    }
}
```

Note: The StructField offset points to the start of the array (first pointer slot). The total_size of the struct increases by `size * 8`. The `compile_model_constructor` needs to know the array size and element constructor to generate the loop.

- [ ] **Step 2: Generate constructor loop for FixedArray fields**

In `compile_model_constructor()`, when processing a LayerDecl whose type_ann is FixedArray:

```rust
// Instead of: compile init expr, store single value
// Do: generate loop that calls init expr N times, storing each result

let array_size = /* extract from type_ann */;
let field_offset = /* from layout */;

// Create loop blocks
let loop_header = builder.create_block();
let loop_body = builder.create_block();
let loop_exit = builder.create_block();

// Init counter
let counter_var = state.new_variable();
builder.declare_var(counter_var, cl_types::I64);
builder.def_var(counter_var, builder.ins().iconst(cl_types::I64, 0));

builder.ins().jump(loop_header, &[]);

// Header: counter < array_size?
builder.switch_to_block(loop_header);
let counter = builder.use_var(counter_var);
let limit = builder.ins().iconst(cl_types::I64, array_size);
let cond = builder.ins().icmp(IntCC::SignedLessThan, counter, limit);
builder.ins().brif(cond, loop_body, &[], loop_exit, &[]);

// Body: call init expr, store at offset
builder.switch_to_block(loop_body);
builder.seal_block(loop_body);
let init_val = self.compile_expr(builder, state, init_expr)?;
let counter = builder.use_var(counter_var);
let elem_offset = builder.ins().imul(counter, builder.ins().iconst(cl_types::I64, 8));
let base_offset = builder.ins().iconst(cl_types::I64, field_offset as i64);
let total_offset = builder.ins().iadd(base_offset, elem_offset);
let addr = builder.ins().iadd(ptr, total_offset);
builder.ins().store(MemFlags::trusted(), init_val, addr, 0);

// Increment
let next = builder.ins().iadd(counter, builder.ins().iconst(cl_types::I64, 1));
builder.def_var(counter_var, next);
builder.ins().jump(loop_header, &[]);

builder.seal_block(loop_header);
builder.switch_to_block(loop_exit);
builder.seal_block(loop_exit);
```

- [ ] **Step 3: Write test**

Create `tests/m18_model_array_test.nsl`:
```nsl
model Layer(dim: int):
    w: Tensor = randn([dim, dim])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

model Stack(dim: int):
    layers: [Layer; 3] = Layer(dim)

    fn forward(self, x: Tensor) -> Tensor:
        # For now, just access first layer to verify construction
        return self.layers  # Will refine once for-loop works

let m = Stack(4)
print("model array construction: PASS")
```

Run: `cargo run -p nsl-cli -- run tests/m18_model_array_test.nsl`
Expected: Construction succeeds, prints PASS

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/compiler.rs tests/m18_model_array_test.nsl
git commit -m "feat(m18a): FixedArray layout computation + constructor loop codegen"
```

---

### Task 16: Codegen — for-loop over model arrays

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

- [ ] **Step 1: Detect FixedModelArray iteration in `compile_for`**

In `compile_for`, before the existing list-based path, check if the iterable's type is `FixedModelArray`:

```rust
let iter_type = self.node_type(iterable.id);
if let Type::FixedModelArray { element_model, size } = iter_type {
    return self.compile_for_model_array(builder, state, pattern, iterable, body, element_model, *size);
}
```

- [ ] **Step 2: Implement `compile_for_model_array`**

```rust
fn compile_for_model_array(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    pattern: &Pattern,
    iterable: &Expr,
    body: &Block,
    element_model: &Symbol,
    size: i64,
) -> Result<(), CodegenError> {
    // Get base address: compile the member access to get self ptr,
    // then compute address of array field
    let base_val = self.compile_expr(builder, state, iterable)?;
    // base_val is the address of the first element in the array

    // Actually, for `self.layers`, compile_member_access returns the
    // value at the field offset, which for an array of pointers is
    // the first pointer. We need the ADDRESS instead.
    // This requires special handling: for FixedModelArray fields,
    // compile_member_access should return the base address (self_ptr + offset)
    // rather than loading the value at that address.

    // Alternative: compile iterable as self_ptr + field_offset directly
    // This means we need to detect the MemberAccess pattern and compute
    // the address rather than loading.

    // For now, simplify: the iterable `self.layers` is a MemberAccess.
    // We can compute self_ptr + layers_offset in the for-loop handler.

    // Pre-declare loop variable
    let loop_var_sym = match &pattern.kind {
        PatternKind::Ident(sym) => *sym,
        _ => return Err(CodegenError::new("only ident patterns in model array for-loops")),
    };
    let elem_var = state.new_variable();
    builder.declare_var(elem_var, cl_types::I64);
    builder.def_var(elem_var, builder.ins().iconst(cl_types::I64, 0));
    state.variables.insert(loop_var_sym, (elem_var, cl_types::I64));

    // Counter
    let counter_var = state.new_variable();
    builder.declare_var(counter_var, cl_types::I64);
    builder.def_var(counter_var, builder.ins().iconst(cl_types::I64, 0));

    let header = builder.create_block();
    let body_block = builder.create_block();
    let increment = builder.create_block();
    let exit = builder.create_block();

    builder.ins().jump(header, &[]);

    // Header
    builder.switch_to_block(header);
    let counter = builder.use_var(counter_var);
    let limit = builder.ins().iconst(cl_types::I64, size);
    let cond = builder.ins().icmp(IntCC::SignedLessThan, counter, limit);
    builder.ins().brif(cond, body_block, &[], exit, &[]);

    // Body: load element pointer from array
    builder.switch_to_block(body_block);
    builder.seal_block(body_block);
    let counter = builder.use_var(counter_var);
    let elem_offset = builder.ins().imul(counter, builder.ins().iconst(cl_types::I64, 8));
    let addr = builder.ins().iadd(base_val, elem_offset);
    let elem_ptr = builder.ins().load(cl_types::I64, MemFlags::trusted(), addr, 0);
    builder.def_var(elem_var, elem_ptr);

    // Compile body
    state.loop_stack.push(LoopContext { continue_block: increment, exit_block: exit });
    for s in &body.stmts {
        self.compile_stmt(builder, state, s)?;
    }
    state.loop_stack.pop();

    if !is_block_filled(builder, body_block) {
        builder.ins().jump(increment, &[]);
    }

    // Increment
    builder.switch_to_block(increment);
    builder.seal_block(increment);
    let counter = builder.use_var(counter_var);
    let next = builder.ins().iadd(counter, builder.ins().iconst(cl_types::I64, 1));
    builder.def_var(counter_var, next);
    builder.ins().jump(header, &[]);

    builder.seal_block(header);
    builder.switch_to_block(exit);
    builder.seal_block(exit);

    Ok(())
}
```

- [ ] **Step 3: Handle FixedModelArray member access in expr.rs**

When `compile_member_access` encounters a field of type FixedModelArray, it should return the address of the array start (self_ptr + offset) instead of loading a value:

```rust
// In compile_member_access, after loading the field:
// Check if this field is a FixedModelArray
if /* field is FixedModelArray */ {
    // Return address, not loaded value
    let offset_val = builder.ins().iconst(cl_types::I64, field.offset as i64);
    return Ok(builder.ins().iadd(obj_val, offset_val));
}
```

Also: when a for-loop body calls `layer.forward(x)` where `layer` is bound to a model pointer from the array, the type system needs to know `layer` is a Model. Store the element model name so that `compile_model_method_call` can resolve it.

- [ ] **Step 4: Update test and run**

Update `tests/m18_model_array_test.nsl`:
```nsl
model Layer(dim: int):
    w: Tensor = randn([dim, dim])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

model Stack(dim: int):
    layers: [Layer; 3] = Layer(dim)

    fn forward(self, x: Tensor) -> Tensor:
        let h = x
        for layer in self.layers:
            h = layer.forward(h)
        return h

let m = Stack(4)
let x = ones([2, 4])
let y = m.forward(x)
print(y)
print("model array for-loop: PASS")
```

Run: `cargo run -p nsl-cli -- run tests/m18_model_array_test.nsl`

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/src/expr.rs tests/m18_model_array_test.nsl
git commit -m "feat(m18a): for-loop iteration over fixed-size model arrays"
```

---

## Chunk 5: Stdlib + End-to-End Test

### Task 17: Rewrite stdlib attention.nsl

**Files:**
- Modify: `stdlib/nsl/nn/attention.nsl`

- [ ] **Step 1: Rewrite with multi-head attention + causal masking**

Replace contents of `stdlib/nsl/nn/attention.nsl` with the code from spec Section 4.1:
```nsl
model Attention(dim: int, num_heads: int):
    q_proj: Tensor = randn([dim, dim])
    k_proj: Tensor = randn([dim, dim])
    v_proj: Tensor = randn([dim, dim])
    out_proj: Tensor = randn([dim, dim])
    _num_heads: Tensor = full([1], float(num_heads))
    _head_dim: Tensor = full([1], float(dim / num_heads))
    _dim: Tensor = full([1], float(dim))

    fn forward(self, x: Tensor) -> Tensor:
        let seq_len = x.shape(0)
        let nh = int(item(self._num_heads))
        let hd = int(item(self._head_dim))
        let d = int(item(self._dim))

        let q = x @ self.q_proj
        let k = x @ self.k_proj
        let v = x @ self.v_proj

        let q = q.reshape([seq_len, nh, hd]).transpose(0, 1)
        let k = k.reshape([seq_len, nh, hd]).transpose(0, 1)
        let v = v.reshape([seq_len, nh, hd]).transpose(0, 1)

        let scale = sqrt(full([1], float(hd)))
        let scores = (q @ k.transpose(-2, -1)) / scale

        let mask = causal_mask(seq_len)
        scores = scores + mask

        let attn = softmax(scores, -1)
        let out = attn @ v

        let out = out.transpose(0, 1).reshape([seq_len, d])
        return out @ self.out_proj
```

- [ ] **Step 2: Test standalone attention**

Create `tests/m18_attention_test.nsl`:
```nsl
from nsl.nn.attention import Attention

let attn = Attention(32, 4)
let x = randn([8, 32])
let y = attn.forward(x)
print(y)
print("attention: PASS")
```

Run: `cargo run -p nsl-cli -- run tests/m18_attention_test.nsl`

- [ ] **Step 3: Commit**

```bash
git add stdlib/nsl/nn/attention.nsl tests/m18_attention_test.nsl
git commit -m "feat(m18a): rewrite attention.nsl with multi-head + causal masking"
```

---

### Task 18: Create stdlib position.nsl and transformer.nsl

**Files:**
- Create: `stdlib/nsl/nn/position.nsl`
- Create: `stdlib/nsl/nn/transformer.nsl`

- [ ] **Step 1: Write position.nsl**

```nsl
model PositionalEmbedding(max_seq: int, dim: int):
    pe: Tensor = randn([max_seq, dim])

    fn forward(self, x: Tensor) -> Tensor:
        let seq_len = x.shape(0)
        let pos = self.pe.slice(0, 0, seq_len)
        return x + pos
```

- [ ] **Step 2: Write transformer.nsl**

```nsl
from nsl.nn.layers import Linear, MLP
from nsl.nn.norms import LayerNorm
from nsl.nn.attention import Attention

model TransformerBlock(dim: int, num_heads: int):
    norm1: LayerNorm = LayerNorm(dim)
    attn: Attention = Attention(dim, num_heads)
    norm2: LayerNorm = LayerNorm(dim)
    mlp: MLP = MLP(dim, dim * 4, dim)

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + self.attn.forward(self.norm1.forward(x))
        return h + self.mlp.forward(self.norm2.forward(h))
```

Note: The `Transformer` model with `[TransformerBlock; 6]` should be defined in the end-to-end test, not stdlib, since the array size is hardcoded.

- [ ] **Step 3: Test transformer block**

Create `tests/m18_transformer_block_test.nsl`:
```nsl
from nsl.nn.transformer import TransformerBlock

let block = TransformerBlock(32, 4)
let x = randn([8, 32])
let y = block.forward(x)
print(y)
print("transformer block: PASS")
```

Run: `cargo run -p nsl-cli -- run tests/m18_transformer_block_test.nsl`

- [ ] **Step 4: Commit**

```bash
git add stdlib/nsl/nn/position.nsl stdlib/nsl/nn/transformer.nsl tests/m18_transformer_block_test.nsl
git commit -m "feat(m18a): add position.nsl and transformer.nsl stdlib modules"
```

---

### Task 19: End-to-end transformer test

**Files:**
- Create: `examples/m18_transformer.nsl`

- [ ] **Step 1: Write end-to-end test program**

Use the test from spec Section 5.1 (the content from the design spec).

- [ ] **Step 2: Run forward pass test**

Run: `cargo run -p nsl-cli -- run examples/m18_transformer.nsl`
Expected: Forward pass produces output tensor with correct shapes, training loss decreases over 3 epochs, prints "m18_transformer: ALL PASS"

- [ ] **Step 3: Commit**

```bash
git add examples/m18_transformer.nsl
git commit -m "feat(m18a): add end-to-end transformer test"
```

---

### Task 20: Final regression check + docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run ALL existing tests**

```bash
cargo test --workspace
cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl
cargo run -p nsl-cli -- test tests/m15_test.nsl
cargo run -p nsl-cli -- run examples/m15_tiny_lm.nsl
cargo run -p nsl-cli -- run examples/m16_quantize.nsl
```

All must pass.

- [ ] **Step 2: Run all M18a tests**

```bash
cargo run -p nsl-cli -- run tests/m18_unsqueeze_test.nsl
cargo run -p nsl-cli -- run tests/m18_select_test.nsl
cargo run -p nsl-cli -- run tests/m18_stack_test.nsl
cargo run -p nsl-cli -- run tests/m18_expand_test.nsl
cargo run -p nsl-cli -- run tests/m18_causal_mask_test.nsl
cargo run -p nsl-cli -- run tests/m18_unsqueeze_grad_test.nsl
cargo run -p nsl-cli -- run tests/m18_expand_grad_test.nsl
cargo run -p nsl-cli -- run tests/m18_stack_grad_test.nsl
cargo run -p nsl-cli -- run tests/m18_nested_model_test.nsl
cargo run -p nsl-cli -- run tests/m18_model_array_test.nsl
cargo run -p nsl-cli -- run tests/m18_attention_test.nsl
cargo run -p nsl-cli -- run tests/m18_transformer_block_test.nsl
cargo run -p nsl-cli -- run examples/m18_transformer.nsl
```

- [ ] **Step 3: Update README.md**

Update the milestone table:
```
| M18a | Transformer foundations (tensor ops + model composition) | Complete |
| M18b | Interop (PyTorch, HuggingFace, ONNX) | Planned |
```

Add M18a test commands to the testing section.

- [ ] **Step 4: Final commit**

```bash
git add README.md
git commit -m "docs: update README for M18a completion"
```
