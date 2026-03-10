# M14: Training DSL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the train block end-to-end with optimizers, schedulers, loss functions, and checkpointing — enabling real model training in NSL.

**Architecture:** The compiler generates training loop code from the existing parsed TrainBlock AST. Optimizers, schedulers, and losses are pure NSL stdlib functions imported via the M13 import system. New Rust runtime primitives provide in-place tensor mutation, element-wise math ops, and dimensional reductions with autodiff tape support.

**Tech Stack:** Rust (runtime ops, checkpoint I/O), Cranelift 0.116 (codegen), NSL (stdlib optimizers/schedulers/losses)

---

## Current State

**What exists:**
- Lexer: train keyword (TokenKind::Train)
- Parser: parse_train_block_stmt() in crates/nsl-parser/src/block.rs:8-119 — parses all sections (data, optimizer, scheduler, step, eval, callbacks, distribute)
- AST: TrainBlock { config, sections, span } in crates/nsl-ast/src/block.rs:9-13, TrainSection enum (lines 16-31), CallbackDef (lines 33-38)
- Semantic: check_train_block() is a no-op at crates/nsl-semantic/src/checker.rs:607-612
- Codegen: StmtKind::TrainBlock(_) is skipped at crates/nsl-codegen/src/stmt.rs:186 and crates/nsl-codegen/src/compiler.rs:1032
- Runtime: Tape-based autodiff with 11 TapeOp variants at crates/nsl-runtime/src/autodiff.rs:25-37
- Runtime: Global nsl_tensor_sum() and nsl_tensor_mean() at crates/nsl-runtime/src/tensor.rs:637-674 (no dim/keepdim support)
- Runtime: Scalar math (sqrt, log, exp, cos, abs) at crates/nsl-runtime/src/math.rs:1-33
- Runtime: NslList at crates/nsl-runtime/src/list.rs:6-10
- Stdlib: Only stdlib/nsl/math.nsl exists
- Import system: Fully functional (M13) — stdlib resolution, name mangling, multi-file compilation
- Grad blocks: Full codegen at crates/nsl-codegen/src/stmt.rs:799-870+
- Existing tests: 17 integration tests with expected output in tests/expected/

**What M14 adds:**
1. Runtime: Element-wise tensor ops (exp, log, sqrt, abs, sign, clamp) with autodiff tape support
2. Runtime: Dimensional reductions (sum/mean with dim+keepdim, reduce_max, gather)
3. Runtime: In-place mutation ops (copy_data, add_inplace, zero_inplace, zeros_like)
4. Runtime: Gradient clipping, checkpoint I/O, scalar cos/floor
5. Semantic: Train block validation
6. Codegen: Full train block compilation (epoch loop, implicit grad, accumulation, optimizer dispatch, scheduler, callbacks, eval)
7. Stdlib: 6 optimizers, 7 schedulers, 4 loss functions (all pure NSL)

**Design doc:** docs/plans/2026-03-10-m14-training-dsl-design.md

---

## Task 1: Runtime — Element-wise Tensor Ops (exp, log, sqrt)

**Files:**
- Modify: crates/nsl-runtime/src/tensor.rs (after line 674)
- Modify: crates/nsl-runtime/src/autodiff.rs (TapeOp enum at line 25, backward match at line 191)

**Step 1: Add new TapeOp variants to autodiff.rs**

In crates/nsl-runtime/src/autodiff.rs, add these variants to the TapeOp enum (after MeanReduce at line 36):

```rust
    Exp { a: i64, out: i64, saved_out: i64 },
    Log { a: i64, out: i64, saved_a: i64 },
    Sqrt { a: i64, out: i64, saved_out: i64 },
```

**Step 2: Add backward pass implementations**

In nsl_tape_backward(), inside the match on op (around line 191), add arms for each new variant. Use the existing pattern from how Mul (around line 220) does it:

```rust
TapeOp::Exp { a, out, saved_out } => {
    if let Some(&g) = grad_map.get(&out) {
        // d/dx exp(x) = exp(x) = saved_output
        let grad_a = nsl_tensor_mul(g, saved_out);
        accumulate_grad(&mut grad_map, a, grad_a);
    }
}
TapeOp::Log { a, out, saved_a } => {
    if let Some(&g) = grad_map.get(&out) {
        // d/dx log(x) = 1/x
        let grad_a = nsl_tensor_div(g, saved_a);
        accumulate_grad(&mut grad_map, a, grad_a);
    }
}
TapeOp::Sqrt { a, out, saved_out } => {
    if let Some(&g) = grad_map.get(&out) {
        // d/dx sqrt(x) = 1/(2*sqrt(x)) = 1/(2*out)
        let two = create_scalar_tensor(2.0);
        let denom = nsl_tensor_mul(two, saved_out);
        let grad_a = nsl_tensor_div(g, denom);
        accumulate_grad(&mut grad_map, a, grad_a);
    }
}
```

**Step 3: Implement element-wise tensor ops in tensor.rs**

Add after nsl_tensor_mean (after line 674). Follow the exact same allocation pattern used by existing ops like nsl_tensor_add (look around line 430) — allocate result data, clone shape/strides, build NslTensor, return pointer as i64:

```rust
// === Element-wise math ops ===

#[no_mangle]
pub extern "C" fn nsl_tensor_exp(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let result_data = alloc_f64(tensor.len as usize);
    for i in 0..tensor.len as usize {
        unsafe { *result_data.add(i) = (*tensor.data.add(i)).exp(); }
    }
    let result = create_tensor_like(tensor_ptr, result_data);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Exp {
            a: tensor_ptr, out: result, saved_out: result,
        });
        NslTensor::from_ptr(result).refcount += 1;
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_log(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let result_data = alloc_f64(tensor.len as usize);
    for i in 0..tensor.len as usize {
        unsafe { *result_data.add(i) = (*tensor.data.add(i)).ln(); }
    }
    let result = create_tensor_like(tensor_ptr, result_data);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Log {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sqrt(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let result_data = alloc_f64(tensor.len as usize);
    for i in 0..tensor.len as usize {
        unsafe { *result_data.add(i) = (*tensor.data.add(i)).sqrt(); }
    }
    let result = create_tensor_like(tensor_ptr, result_data);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Sqrt {
            a: tensor_ptr, out: result, saved_out: result,
        });
        NslTensor::from_ptr(result).refcount += 1;
    }
    result
}
```

Note: You will likely need to add a create_tensor_like(src_ptr, new_data) helper that allocates a new NslTensor with the same shape/strides as src but with new data. If no such helper exists, create one — it clones the shape and strides arrays from the source tensor.

**Step 4: Build and verify compilation**

Run: `cargo build -p nsl-runtime`
Expected: Compiles with no errors.

**Step 5: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-runtime/src/autodiff.rs
git commit -m "feat(m14): add element-wise exp, log, sqrt tensor ops with autodiff"
```

---

## Task 2: Runtime — Element-wise abs, sign, clamp + Tape Support

**Files:**
- Modify: crates/nsl-runtime/src/tensor.rs
- Modify: crates/nsl-runtime/src/autodiff.rs

**Step 1: Add TapeOp variants**

In autodiff.rs TapeOp enum, add:

```rust
    Abs { a: i64, out: i64, saved_a: i64 },
    Clamp { a: i64, out: i64, saved_a: i64, min_val: f64, max_val: f64 },
```

Note: sign is non-differentiable so no tape entry needed.

**Step 2: Add backward passes**

```rust
TapeOp::Abs { a, out, saved_a } => {
    if let Some(&g) = grad_map.get(&out) {
        let sign_a = nsl_tensor_sign(saved_a);
        let grad_a = nsl_tensor_mul(g, sign_a);
        accumulate_grad(&mut grad_map, a, grad_a);
    }
}
TapeOp::Clamp { a, out, saved_a, min_val, max_val } => {
    if let Some(&g) = grad_map.get(&out) {
        // Gradient passes through where input was unclamped, zero where clamped
        let sa = NslTensor::from_ptr(saved_a);
        let grad_tensor = NslTensor::from_ptr(g);
        let result_data = alloc_f64(sa.len as usize);
        for i in 0..sa.len as usize {
            let val = unsafe { *sa.data.add(i) };
            let g_val = unsafe { *grad_tensor.data.add(i) };
            if val <= min_val || val >= max_val {
                unsafe { *result_data.add(i) = 0.0; }
            } else {
                unsafe { *result_data.add(i) = g_val; }
            }
        }
        let grad_a = create_tensor_like(saved_a, result_data);
        accumulate_grad(&mut grad_map, a, grad_a);
    }
}
```

**Step 3: Implement ops in tensor.rs**

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_abs(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let result_data = alloc_f64(tensor.len as usize);
    for i in 0..tensor.len as usize {
        unsafe { *result_data.add(i) = (*tensor.data.add(i)).abs(); }
    }
    let result = create_tensor_like(tensor_ptr, result_data);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Abs {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sign(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let result_data = alloc_f64(tensor.len as usize);
    for i in 0..tensor.len as usize {
        let val = unsafe { *tensor.data.add(i) };
        unsafe {
            *result_data.add(i) = if val > 0.0 { 1.0 } else if val < 0.0 { -1.0 } else { 0.0 };
        }
    }
    create_tensor_like(tensor_ptr, result_data)
    // No tape recording for sign (non-differentiable)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_clamp(tensor_ptr: i64, min_val: f64, max_val: f64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let result_data = alloc_f64(tensor.len as usize);
    for i in 0..tensor.len as usize {
        let val = unsafe { *tensor.data.add(i) };
        unsafe { *result_data.add(i) = val.max(min_val).min(max_val); }
    }
    let result = create_tensor_like(tensor_ptr, result_data);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Clamp {
            a: tensor_ptr, out: result, saved_a: tensor_ptr, min_val, max_val,
        });
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
    }
    result
}
```

**Step 4: Build**

Run: `cargo build -p nsl-runtime`
Expected: Compiles with no errors.

**Step 5: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-runtime/src/autodiff.rs
git commit -m "feat(m14): add abs, sign, clamp tensor ops with autodiff"
```

---

## Task 3: Runtime — Dimensional Reductions (sum/mean with dim+keepdim)

**Files:**
- Modify: crates/nsl-runtime/src/tensor.rs (lines 637-674)
- Modify: crates/nsl-runtime/src/autodiff.rs

This task upgrades existing nsl_tensor_sum and nsl_tensor_mean to support dim and keepdim parameters, while keeping backward compatibility (dim=-1 means global reduction).

**Step 1: Update TapeOp variants**

Replace existing SumReduce and MeanReduce in autodiff.rs:

```rust
    SumReduce { a: i64, out: i64, dim: i64, keepdim: bool, input_shape: Vec<i64> },
    MeanReduce { a: i64, out: i64, dim: i64, keepdim: bool, num_elements: i64, input_shape: Vec<i64> },
```

**Step 2: Keep old function signatures as wrappers**

Keep nsl_tensor_sum(tensor_ptr) calling nsl_tensor_sum_dim(tensor_ptr, -1, 0) for backward compat. Add new:

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_sum_dim(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let keepdim_bool = keepdim != 0;

    if dim == -1 {
        // Global reduction (backward compat)
        let mut total = 0.0;
        for i in 0..tensor.len as usize {
            total += unsafe { *tensor.data.add(i) };
        }
        let result = create_scalar_tensor(total);
        if autodiff::is_recording() {
            let shape = get_shape_vec(tensor_ptr);
            autodiff::maybe_record(autodiff::TapeOp::SumReduce {
                a: tensor_ptr, out: result, dim: -1, keepdim: false, input_shape: shape,
            });
        }
        return result;
    }

    let actual_dim = if dim < 0 { tensor.ndim + dim } else { dim } as usize;
    let dim_size = unsafe { *tensor.shape.add(actual_dim) } as usize;

    // Compute output shape
    let mut out_shape = Vec::new();
    for d in 0..tensor.ndim as usize {
        if d == actual_dim {
            if keepdim_bool { out_shape.push(1i64); }
        } else {
            out_shape.push(unsafe { *tensor.shape.add(d) });
        }
    }

    let out_len: i64 = out_shape.iter().product();
    let result_data = alloc_f64(out_len as usize);

    // Reduction: for each output position, sum over the reduction dimension
    // Use stride-based indexing to iterate correctly
    // The key: for a given output multi-index, expand it to input multi-index
    // by inserting all values 0..dim_size at the reduction dimension position
    for out_idx in 0..out_len as usize {
        let mut total = 0.0;
        // Convert flat out_idx to multi-index, expand along dim, sum
        // ... (implement using strides or nested iteration) ...
        unsafe { *result_data.add(out_idx) = total; }
    }

    let result = create_tensor_with_shape(result_data, &out_shape);
    if autodiff::is_recording() {
        let shape = get_shape_vec(tensor_ptr);
        autodiff::maybe_record(autodiff::TapeOp::SumReduce {
            a: tensor_ptr, out: result, dim, keepdim: keepdim_bool, input_shape: shape,
        });
    }
    result
}
```

**Step 3: Update backward for SumReduce**

For global (dim=-1): gradient is broadcast to all elements (existing behavior with input_shape info).
For dimensional: gradient is broadcast along the reduced dimension — repeat grad values along dim.

**Step 4: Do the same for mean_dim**

Same pattern as sum but divides by dim_size.

**Step 5: Build and verify backward compat**

Run: `cargo build -p nsl-cli && cargo run -p nsl-cli -- run examples/m12_grad_basic.nsl`
Expected: Same output as before (global reductions still work).

**Step 6: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-runtime/src/autodiff.rs
git commit -m "feat(m14): add dimensional sum/mean reductions with keepdim support"
```

---

## Task 4: Runtime — reduce_max and gather with Tape Support

**Files:**
- Modify: crates/nsl-runtime/src/tensor.rs
- Modify: crates/nsl-runtime/src/autodiff.rs

**Step 1: Add TapeOp variants**

```rust
    ReduceMax { a: i64, out: i64, dim: i64, keepdim: bool, saved_argmax: Vec<usize> },
    Gather { a: i64, out: i64, dim: i64, indices_ptr: i64 },
```

**Step 2: Implement reduce_max**

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_max(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let keepdim_bool = keepdim != 0;
    let actual_dim = if dim < 0 { tensor.ndim + dim } else { dim } as usize;
    let dim_size = unsafe { *tensor.shape.add(actual_dim) } as usize;

    // Compute output shape (same logic as sum_dim)
    // For each output position, find max along actual_dim, track argmax for backward
    // Store argmax indices in a Vec for the tape

    // ... implementation ...

    // Record tape with saved_argmax
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::ReduceMax {
            a: tensor_ptr, out: result, dim, keepdim: keepdim_bool, saved_argmax: argmax_indices,
        });
    }
    result
}
```

Backward: gradient flows only to the argmax positions (scatter grad to max indices, zero elsewhere).

**Step 3: Implement gather**

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_gather(tensor_ptr: i64, dim: i64, indices_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let indices = NslTensor::from_ptr(indices_ptr);
    // For cross_entropy: tensor is [batch, classes], dim=1, indices is [batch]
    // Output is [batch]: output[b] = tensor[b, indices[b]]

    let actual_dim = if dim < 0 { tensor.ndim + dim } else { dim } as usize;
    let out_len = indices.len;
    let result_data = alloc_f64(out_len as usize);

    // For 2D case (most common): iterate batch, index into dim
    for i in 0..out_len as usize {
        let idx = unsafe { *indices.data.add(i) } as usize;
        // Compute linear offset: i * stride[0] + idx * stride[1]
        let offset = i * (unsafe { *tensor.strides.add(0) } as usize)
                   + idx * (unsafe { *tensor.strides.add(1) } as usize);
        unsafe { *result_data.add(i) = *tensor.data.add(offset); }
    }

    let result = create_tensor_1d(result_data, out_len);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Gather {
            a: tensor_ptr, out: result, dim, indices_ptr,
        });
        NslTensor::from_ptr(indices_ptr).refcount += 1;
    }
    result
}
```

Backward for Gather: scatter upstream gradient to the gathered positions (inverse of gather).

**Step 4: Build**

Run: `cargo build -p nsl-runtime`
Expected: Compiles with no errors.

**Step 5: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-runtime/src/autodiff.rs
git commit -m "feat(m14): add reduce_max and gather tensor ops with autodiff"
```

---

## Task 5: Runtime — In-place Mutation Ops + zeros_like

**Files:**
- Modify: crates/nsl-runtime/src/tensor.rs

These ops are NOT taped. They are used by optimizer steps outside grad blocks.

**Step 1: Implement copy_data, add_inplace, zero_inplace, zeros_like**

```rust
/// Copy src data into dst tensor. Asserts same shape. NOT taped.
#[no_mangle]
pub extern "C" fn nsl_tensor_copy_data(dst_ptr: i64, src_ptr: i64) {
    let dst = NslTensor::from_ptr_mut(dst_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    assert_eq!(dst.len, src.len, "copy_data: tensors must have same length");
    unsafe {
        std::ptr::copy_nonoverlapping(src.data, dst.data, dst.len as usize);
    }
}

/// In-place addition: dst[i] += src[i]. For gradient accumulation.
#[no_mangle]
pub extern "C" fn nsl_tensor_add_inplace(dst_ptr: i64, src_ptr: i64) {
    let dst = NslTensor::from_ptr_mut(dst_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    assert_eq!(dst.len, src.len, "add_inplace: tensors must have same length");
    for i in 0..dst.len as usize {
        unsafe { *dst.data.add(i) += *src.data.add(i); }
    }
}

/// Zero all elements. For resetting gradient accumulators.
#[no_mangle]
pub extern "C" fn nsl_tensor_zero_inplace(tensor_ptr: i64) {
    let tensor = NslTensor::from_ptr_mut(tensor_ptr);
    for i in 0..tensor.len as usize {
        unsafe { *tensor.data.add(i) = 0.0; }
    }
}

/// Allocate new tensor with same shape, filled with zeros.
#[no_mangle]
pub extern "C" fn nsl_tensor_zeros_like(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape_list = crate::list::nsl_list_new();
    for i in 0..tensor.ndim as usize {
        let dim = unsafe { *tensor.shape.add(i) };
        crate::list::nsl_list_push(shape_list, dim);
    }
    nsl_tensor_zeros(shape_list)
}
```

Note: You may need to add a from_ptr_mut helper on NslTensor that returns a mutable reference. Check if it exists. If not, create one by casting the i64 pointer.

**Step 2: Build**

Run: `cargo build -p nsl-runtime`
Expected: Compiles with no errors.

**Step 3: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs
git commit -m "feat(m14): add in-place tensor ops (copy_data, add_inplace, zero_inplace, zeros_like)"
```

---

## Task 6: Runtime — Gradient Clipping + Scalar floor

**Files:**
- Modify: crates/nsl-runtime/src/tensor.rs (gradient clipping)
- Modify: crates/nsl-runtime/src/math.rs (floor — cos already exists at line 13)

**Step 1: Implement gradient norm clipping**

Add to tensor.rs:

```rust
/// Compute global gradient norm and clip if exceeds max_norm.
/// grad_list is a NslList of tensor pointers.
#[no_mangle]
pub extern "C" fn nsl_clip_grad_norm(grad_list_ptr: i64, max_norm: f64) {
    let list = crate::list::NslList::from_ptr(grad_list_ptr);
    let mut total_sq = 0.0;
    for i in 0..list.len as usize {
        let tensor_ptr = unsafe { *list.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        for j in 0..tensor.len as usize {
            let val = unsafe { *tensor.data.add(j) };
            total_sq += val * val;
        }
    }
    let total_norm = total_sq.sqrt();
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for i in 0..list.len as usize {
            let tensor_ptr = unsafe { *list.data.add(i) };
            let tensor = NslTensor::from_ptr_mut(tensor_ptr);
            for j in 0..tensor.len as usize {
                unsafe { *tensor.data.add(j) *= scale; }
            }
        }
    }
}
```

**Step 2: Add floor to math.rs**

In crates/nsl-runtime/src/math.rs, add after the last function:

```rust
#[no_mangle]
pub extern "C" fn nsl_floor(x: f64) -> f64 {
    x.floor()
}
```

Note: nsl_cos already exists at line 13.

**Step 3: Build**

Run: `cargo build -p nsl-runtime`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-runtime/src/math.rs
git commit -m "feat(m14): add gradient clipping and floor scalar op"
```

---

## Task 7: Runtime — Checkpoint I/O (.nslm format)

**Files:**
- Create: crates/nsl-runtime/src/checkpoint.rs
- Modify: crates/nsl-runtime/src/lib.rs (add pub mod checkpoint; after line 20)

**Step 1: Create checkpoint.rs**

Create crates/nsl-runtime/src/checkpoint.rs with save and load functions:

```rust
use crate::tensor::NslTensor;
use crate::list::NslList;
use std::io::Write;

const MAGIC: &[u8; 4] = b"NSLM";
const VERSION: u32 = 1;

/// Save model parameters to .nslm binary format.
/// path_ptr/path_len: string pointer and length for file path
/// param_names_ptr: NslList of string pointers
/// param_tensors_ptr: NslList of tensor pointers
#[no_mangle]
pub extern "C" fn nsl_model_save(
    path_ptr: i64, path_len: i64,
    param_names_ptr: i64,
    param_tensors_ptr: i64,
) {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    let names = NslList::from_ptr(param_names_ptr);
    let tensors = NslList::from_ptr(param_tensors_ptr);
    assert_eq!(names.len, tensors.len, "model_save: name/tensor count mismatch");

    // Build JSON header
    let mut params_json = Vec::new();
    let mut data_offset: u64 = 0;
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        assert_tensor_contiguous(tensor, i);
        let nbytes = (tensor.len as u64) * 8;
        let shape: Vec<i64> = (0..tensor.ndim as usize)
            .map(|d| unsafe { *tensor.shape.add(d) })
            .collect();
        params_json.push(format!(
            r#"{{"name":"param_{}","shape":{:?},"dtype":"f64","offset":{},"nbytes":{}}}"#,
            i, shape, data_offset, nbytes
        ));
        data_offset += nbytes;
    }
    let header = format!(r#"{{"params":[{}]}}"#, params_json.join(","));
    let header_bytes = header.as_bytes();

    let mut file = std::fs::File::create(path).expect("model_save: cannot create file");
    file.write_all(MAGIC).unwrap();
    file.write_all(&VERSION.to_le_bytes()).unwrap();
    file.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    file.write_all(header_bytes).unwrap();

    // Pad to 64-byte alignment
    let total_header = 4 + 4 + 8 + header_bytes.len();
    let padding = (64 - (total_header % 64)) % 64;
    file.write_all(&vec![0u8; padding]).unwrap();

    // Raw tensor data (little-endian f64)
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        for j in 0..tensor.len as usize {
            let val = unsafe { *tensor.data.add(j) };
            file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
}

/// Load model parameters from .nslm binary format into existing tensors.
#[no_mangle]
pub extern "C" fn nsl_model_load(
    path_ptr: i64, path_len: i64,
    param_tensors_ptr: i64,
) {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    let tensors = NslList::from_ptr(param_tensors_ptr);
    let data = std::fs::read(path).expect("model_load: cannot read file");

    assert_eq!(&data[0..4], MAGIC, "model_load: invalid .nslm file");
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    assert_eq!(version, VERSION, "model_load: unsupported version");
    let header_size = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;

    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;

    let mut offset = data_start;
    for i in 0..tensors.len as usize {
        let tensor_ptr = unsafe { *tensors.data.add(i) };
        let tensor = NslTensor::from_ptr_mut(tensor_ptr);
        for j in 0..tensor.len as usize {
            let val = f64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
            unsafe { *tensor.data.add(j) = val; }
            offset += 8;
        }
    }
}

fn assert_tensor_contiguous(tensor: &NslTensor, idx: usize) {
    if tensor.ndim <= 1 { return; }
    let mut expected_stride = 1i64;
    for d in (0..tensor.ndim as usize).rev() {
        let actual = unsafe { *tensor.strides.add(d) };
        assert_eq!(
            actual, expected_stride,
            "model_save: parameter {} is not contiguous (dim {} stride {} expected {})",
            idx, d, actual, expected_stride
        );
        expected_stride *= unsafe { *tensor.shape.add(d) };
    }
}
```

**Step 2: Add module to lib.rs**

In crates/nsl-runtime/src/lib.rs, add after pub mod autodiff; (line 20):
```rust
pub mod checkpoint;
```

**Step 3: Build**

Run: `cargo build -p nsl-runtime`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add crates/nsl-runtime/src/checkpoint.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(m14): add .nslm checkpoint save/load"
```

---

## Task 8: Codegen — Register New Runtime Functions

**Files:**
- Modify: crates/nsl-codegen/src/compiler.rs (function declarations)
- Modify: crates/nsl-codegen/src/expr.rs (builtin dispatch)

**Step 1: Declare new runtime functions in compiler.rs**

In the Compiler::new() function (or wherever runtime functions are declared with declare_function), add declarations for all new functions. Follow the exact same pattern used for existing declarations like nsl_tensor_sum and nsl_tensor_zeros:

```
nsl_tensor_exp(i64) -> i64
nsl_tensor_log(i64) -> i64
nsl_tensor_sqrt(i64) -> i64
nsl_tensor_abs(i64) -> i64
nsl_tensor_sign(i64) -> i64
nsl_tensor_clamp(i64, f64, f64) -> i64
nsl_tensor_reduce_max(i64, i64, i64) -> i64
nsl_tensor_gather(i64, i64, i64) -> i64
nsl_tensor_sum_dim(i64, i64, i64) -> i64
nsl_tensor_mean_dim(i64, i64, i64) -> i64
nsl_tensor_copy_data(i64, i64) -> void
nsl_tensor_add_inplace(i64, i64) -> void
nsl_tensor_zero_inplace(i64) -> void
nsl_tensor_zeros_like(i64) -> i64
nsl_clip_grad_norm(i64, f64) -> void
nsl_model_save(i64, i64, i64, i64) -> void
nsl_model_load(i64, i64, i64) -> void
nsl_floor(f64) -> f64
```

**Step 2: Add builtin dispatch in expr.rs**

In compile_call (around line 361 in expr.rs), where builtin function names are matched (around line 525 for "zeros"|"ones"|"rand"), add dispatch:

```rust
"exp" => self.compile_call_by_name(builder, "nsl_tensor_exp", &[args[0]]),
"log" => self.compile_call_by_name(builder, "nsl_tensor_log", &[args[0]]),
"sqrt" => self.compile_call_by_name(builder, "nsl_tensor_sqrt", &[args[0]]),
"abs" => self.compile_call_by_name(builder, "nsl_tensor_abs", &[args[0]]),
"sign" => self.compile_call_by_name(builder, "nsl_tensor_sign", &[args[0]]),
"neg" => self.compile_call_by_name(builder, "nsl_tensor_neg", &[args[0]]),
"copy_data" => {
    self.compile_call_by_name(builder, "nsl_tensor_copy_data", &[args[0], args[1]])?;
    Ok(args[0])
}
"clamp" => self.compile_call_by_name(builder, "nsl_tensor_clamp", &[args[0], args[1], args[2]]),
"reduce_max" => self.compile_call_by_name(builder, "nsl_tensor_reduce_max", &[args[0], args[1], args[2]]),
"gather" => self.compile_call_by_name(builder, "nsl_tensor_gather", &[args[0], args[1], args[2]]),
"zeros_like" => self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[args[0]]),
```

For sum and mean with dim arguments: if called with 3 args (tensor, dim, keepdim), dispatch to nsl_tensor_sum_dim / nsl_tensor_mean_dim. If called with 1 arg, use existing nsl_tensor_sum / nsl_tensor_mean.

**Step 3: Build the full project**

Run: `cargo build -p nsl-cli`
Expected: Compiles with no errors.

**Step 4: Commit**

```bash
git add crates/nsl-codegen/src/compiler.rs crates/nsl-codegen/src/expr.rs
git commit -m "feat(m14): register new runtime functions in codegen"
```

---

## Task 9: Stdlib — Loss Functions

**Files:**
- Create: stdlib/nsl/nn/ directory
- Create: stdlib/nsl/nn/losses.nsl

**Step 1: Create directory and loss functions file**

```bash
mkdir -p stdlib/nsl/nn
```

Create stdlib/nsl/nn/losses.nsl:

```nsl
fn mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    let diff = pred - target
    return mean(diff * diff)

fn l1_loss(pred: Tensor, target: Tensor) -> Tensor:
    return mean(abs(pred - target))

fn cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    let max_val = reduce_max(logits, 1, 1)
    let shifted = logits - max_val
    let exp_sum = sum(exp(shifted), 1, 1)
    let log_probs = shifted - log(exp_sum)
    let nll = neg(gather(log_probs, 1, targets))
    return mean(nll)

fn bce_loss(pred: Tensor, target: Tensor) -> Tensor:
    let clamped = clamp(pred, 0.0000001, 0.9999999)
    return neg(mean(target * log(clamped) + (1.0 - target) * log(1.0 - clamped)))
```

Note: reduce_max/sum/gather use positional args: (tensor, dim, keepdim_as_int). 1 means keepdim=true.

**Step 2: Test mse_loss with a simple program**

Create examples/m14_mse_test.nsl:

```nsl
from nsl.nn.losses import mse_loss

let pred = ones([4])
let target = zeros([4])
let loss = mse_loss(pred, target)
print(loss)
```

Run: `cargo build -p nsl-cli && cargo run -p nsl-cli -- run examples/m14_mse_test.nsl`
Expected: Prints 1 (mean of [1,1,1,1] = 1.0, printed as integer because runtime formats integer-valued floats without decimal)

**Step 3: Create expected output**

Create tests/expected/m14_mse_test.txt:
```
1
```

**Step 4: Commit**

```bash
git add stdlib/nsl/nn/losses.nsl examples/m14_mse_test.nsl tests/expected/m14_mse_test.txt
git commit -m "feat(m14): add loss functions (mse, l1, cross_entropy, bce) to stdlib"
```

---

## Task 10: Stdlib — Schedulers

**Files:**
- Create: stdlib/nsl/optim/ directory
- Create: stdlib/nsl/optim/schedulers.nsl

**Step 1: Create directory and scheduler functions**

```bash
mkdir -p stdlib/nsl/optim
```

Create stdlib/nsl/optim/schedulers.nsl with all 7 scheduler functions as specified in the design doc (constant_lr, step_lr, exponential_lr, linear_decay, cosine_anneal, warmup_cosine, one_cycle).

**Step 2: Test schedulers**

Create examples/m14_scheduler_test.nsl that imports and calls warmup_cosine and constant_lr with known inputs. Verify outputs match expected values.

Run: `cargo build -p nsl-cli && cargo run -p nsl-cli -- run examples/m14_scheduler_test.nsl`

**Step 3: Create expected output and commit**

```bash
git add stdlib/nsl/optim/schedulers.nsl examples/m14_scheduler_test.nsl tests/expected/m14_scheduler_test.txt
git commit -m "feat(m14): add learning rate schedulers to stdlib"
```

---

## Task 11: Stdlib — Optimizers (SGD, Adam, AdamW)

**Files:**
- Create: stdlib/nsl/optim/sgd.nsl
- Create: stdlib/nsl/optim/adam.nsl
- Create: stdlib/nsl/optim/adamw.nsl

**Step 1: Write optimizer functions**

Create each file with the optimizer step function as specified in the design doc. Key points:
- All are void functions (no return value)
- All use copy_data for in-place mutation
- AdamW combines weight decay and gradient update in single copy_data

See design doc section "Stdlib Optimizers" for exact code.

**Step 2: Commit**

```bash
git add stdlib/nsl/optim/sgd.nsl stdlib/nsl/optim/adam.nsl stdlib/nsl/optim/adamw.nsl
git commit -m "feat(m14): add SGD, Adam, AdamW optimizers to stdlib"
```

---

## Task 12: Stdlib — Optimizers (Lion, Muon, SOAP)

**Files:**
- Create: stdlib/nsl/optim/lion.nsl
- Create: stdlib/nsl/optim/muon.nsl
- Create: stdlib/nsl/optim/soap.nsl

**Step 1: Write optimizer functions**

Create each file as specified in the design doc.

**Step 2: Commit**

```bash
git add stdlib/nsl/optim/lion.nsl stdlib/nsl/optim/muon.nsl stdlib/nsl/optim/soap.nsl
git commit -m "feat(m14): add Lion, Muon, SOAP optimizers to stdlib"
```

---

## Task 13: Semantic — Train Block Validation

**Files:**
- Modify: crates/nsl-semantic/src/checker.rs (replace no-op at line 607-612)

**Step 1: Replace the no-op check_train_block**

Replace the empty function at line 607-612 with validation that:
1. Requires model= config arg (check expr type is Model)
2. Requires epochs= config arg
3. Requires optimizer: section
4. Requires step(batch): section
5. Validates optional accumulate=, clip_grad_norm=, precision=
6. Walks step body, eval body, callback bodies through check_stmt
7. Warns if precision is not f64/fp32

See design doc section "Semantic Validation" for the full implementation.

**Step 2: Build**

Run: `cargo build -p nsl-semantic && cargo build -p nsl-cli`
Expected: Compiles with no errors.

**Step 3: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs
git commit -m "feat(m14): implement train block semantic validation"
```

---

## Task 14: Codegen — Train Block Compilation (Core Loop)

This is the largest task. The compiler generates the entire training loop.

**Files:**
- Modify: crates/nsl-codegen/src/stmt.rs (replace no-op at line 186, add compile_train_block)
- Modify: crates/nsl-codegen/src/compiler.rs (remove TrainBlock skip at line 1032)

**Step 1: Remove TrainBlock skip**

In stmt.rs line 186, replace the no-op with a call to compile_train_block.
In compiler.rs line 1032, remove TrainBlock from the filter that skips non-compilable statements.

**Step 2: Implement compile_train_block**

This method needs to:

1. **Extract config** — Parse model, epochs, accumulate (default 1), clip_grad_norm from train.config args
2. **Extract optimizer info** — Identify optimizer type (SGD/Adam/etc) and hyperparams from TrainSection::Optimizer
3. **Extract scheduler info** — Identify scheduler type and params from TrainSection::Scheduler (optional)
4. **Collect model params** — Build NslList of param tensor pointers from the model variable
5. **Create optimizer state** — For each param, call zeros_like to create momentum/variance buffers. Number of buffers depends on optimizer type (SGD=1, Adam/AdamW=2, etc.)
6. **Create accum_grads** — zeros_like for each param
7. **Init lr and step_count** — Local variables
8. **Emit epoch loop** — Create Cranelift blocks for loop header, body, exit. Use icmp + brif for condition.
9. **Inside epoch body:**
   a. Zero accum_grads (call zero_inplace per param)
   b. Init running_loss = 0.0
   c. Accumulation loop (for micro in 0..accumulate):
      - tape_start(param_list)
      - Compile step body statements (reuse compile_stmt)
      - Find the loss variable assignment in step body
      - Scale loss: loss * (1.0/accumulate)
      - running_loss += loss.item()
      - tape_backward(loss, param_list) -> grads_list
      - tape_stop()
      - For each param: add_inplace(accum_grads[i], grads[i])
   d. If clip_grad_norm: call nsl_clip_grad_norm(accum_grads_list, max_norm)
   e. Optimizer step: for each param, call the imported optimizer step function
   f. Zero accum_grads for next epoch
   g. Scheduler: call imported scheduler function to update lr
   h. step_count += 1
   i. Compile callbacks (on_step with step_count and running_loss)
   j. Compile eval block body

**Key integration with import system:** The train block codegen must ensure the optimizer/scheduler stdlib modules are imported. This means adding them to the module dependency graph in the CLI's run_build_multi. The simplest approach: when the CLI detects a train block, automatically add the needed stdlib imports to the import list before compilation.

**Step 3: Build and test**

This is tested in Task 15.

**Step 4: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/src/compiler.rs
git commit -m "feat(m14): implement train block codegen (core loop, implicit grad, optimizer dispatch)"
```

---

## Task 15: CLI — Auto-import Optimizer/Scheduler/Loss Modules for Train Blocks

**Files:**
- Modify: crates/nsl-cli/src/main.rs (run_build_multi function, around line 301)

**Step 1: Detect train blocks and inject stdlib imports**

In run_build_multi, after parsing the entry file, scan for TrainBlock statements. If found, extract the optimizer and scheduler names and automatically add the corresponding stdlib module paths to the import list.

For example, if the train block has `optimizer: AdamW(...)`, automatically add `stdlib/nsl/optim/adamw.nsl` to the dependency list. The M13 import infrastructure handles the rest (resolution, mangling, multi-file compilation).

Similarly for `scheduler: WarmupCosine(...)` -> add `stdlib/nsl/optim/schedulers.nsl`.

For loss functions used in the step body (like mse_loss), the user already writes explicit `from nsl.nn.losses import mse_loss` which M13 handles.

**Step 2: Build**

Run: `cargo build -p nsl-cli`

**Step 3: Commit**

```bash
git add crates/nsl-cli/src/main.rs
git commit -m "feat(m14): auto-import optimizer/scheduler stdlib modules for train blocks"
```

---

## Task 16: Integration — Train Block with SGD (E2E Test)

**Files:**
- Create: examples/m14_sgd_basic.nsl
- Create: tests/expected/m14_sgd_basic.txt

**Step 1: Write the simplest possible training test**

Create examples/m14_sgd_basic.nsl:

```nsl
from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

train(model=m, epochs=5):
    optimizer: SGD(lr=0.01)
    step(batch):
        let pred = m.forward(x)
        loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(loss)
```

**Step 2: Build and run**

Run: `cargo build -p nsl-cli && cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl`
Expected: 5 decreasing loss values printed.

**Step 3: Capture expected output**

Save the output to tests/expected/m14_sgd_basic.txt.

**Step 4: Commit**

```bash
git add examples/m14_sgd_basic.nsl tests/expected/m14_sgd_basic.txt
git commit -m "feat(m14): add basic SGD training integration test"
```

---

## Task 17: Integration — AdamW + Scheduler Test

**Files:**
- Create: examples/m14_adam_scheduler.nsl
- Create: tests/expected/m14_adam_scheduler.txt

**Step 1: Write test with AdamW and WarmupCosine scheduler**

Test trains 10 epochs with AdamW (weight_decay=0.01) and WarmupCosine scheduler, printing loss each step. Loss should decrease.

**Step 2: Build, run, capture expected output, commit**

```bash
git add examples/m14_adam_scheduler.nsl tests/expected/m14_adam_scheduler.txt
git commit -m "feat(m14): add AdamW + scheduler integration test"
```

---

## Task 18: Integration — Gradient Accumulation Test

**Files:**
- Create: examples/m14_grad_accum.nsl
- Create: tests/expected/m14_grad_accum.txt

**Step 1: Write test with accumulate=2**

Test trains with accumulate=2, verifying that training still converges. Loss should decrease.

**Step 2: Build, run, capture expected output, commit**

```bash
git add examples/m14_grad_accum.nsl tests/expected/m14_grad_accum.txt
git commit -m "feat(m14): add gradient accumulation integration test"
```

---

## Task 19: Integration — Checkpoint Save/Load Test

**Files:**
- Create: examples/m14_checkpoint.nsl
- Create: tests/expected/m14_checkpoint.txt

**Step 1: Write test**

Train a model, save checkpoint, create fresh model, load checkpoint, verify predictions match.

**Step 2: Build, run, capture expected output, commit**

```bash
git add examples/m14_checkpoint.nsl tests/expected/m14_checkpoint.txt
git commit -m "feat(m14): add checkpoint save/load integration test"
```

---

## Task 20: Integration — All Optimizers Test

**Files:**
- Create: examples/m14_all_optimizers.nsl
- Create: tests/expected/m14_all_optimizers.txt

**Step 1: Write test exercising all 6 optimizers**

Create sequential training runs with each optimizer (SGD, Adam, AdamW, Lion, Muon, SOAP), each training 3 epochs. Print final loss for each. All should be less than initial loss.

**Step 2: Build, run, capture expected output, commit**

```bash
git add examples/m14_all_optimizers.nsl tests/expected/m14_all_optimizers.txt
git commit -m "feat(m14): add all-optimizers integration test"
```

---

## Task 21: Integration — Callbacks and Eval Test

**Files:**
- Create: examples/m14_callbacks.nsl
- Create: tests/expected/m14_callbacks.txt

**Step 1: Write test with callbacks and eval block**

Test that on_step(step, loss) fires each epoch with correct step number and averaged loss, and eval(epoch) fires with correct epoch number.

**Step 2: Build, run, capture expected output, commit**

```bash
git add examples/m14_callbacks.nsl tests/expected/m14_callbacks.txt
git commit -m "feat(m14): add callback and eval integration test"
```

---

## Task 22: Regression — Run All Existing Tests

**Files:** None modified.

**Step 1: Run all existing M9-M13 tests**

```bash
cargo build -p nsl-cli
cargo run -p nsl-cli -- run examples/hello.nsl
cargo run -p nsl-cli -- run examples/m9_tensors.nsl
cargo run -p nsl-cli -- run examples/m12_grad_basic.nsl
cargo run -p nsl-cli -- run examples/m12_grad_matmul.nsl
cargo run -p nsl-cli -- run examples/m12_grad_model.nsl
cargo run -p nsl-cli -- run examples/m12_no_grad.nsl
cargo run -p nsl-cli -- run examples/m13_stdlib_import.nsl
cargo run -p nsl-cli -- run examples/m13_import_alias.nsl
cargo run -p nsl-cli -- run examples/m13_mixed_imports.nsl
```

For each, compare output against tests/expected/*.txt. All must match.

**Step 2: Run cargo test for unit tests**

```bash
cargo test --workspace
```

Expected: All existing unit tests pass.

**Step 3: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

Expected: No warnings.

---

## Task 23: Final — Update Memory + Clean Up

**Files:**
- Modify: ~/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md

**Step 1: Update MEMORY.md**

Update the Development Roadmap section to mark M14 as complete and add plan reference.

**Step 2: Commit any cleanup**

```bash
git add -A
git commit -m "docs(m14): update memory and project docs"
```
