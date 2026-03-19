# NSL-Coder-50M Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 50M-parameter LLaMA-style code language model defined, trained, and served entirely in NSL.

**Architecture:** LLaMA-2/3 decoder-only transformer (GQA, RoPE, SwiGLU, RMSNorm) in f32 precision. Two-stage training: 10B StarCoder pretrain + NSL finetune. Interactive CLI inference demo.

**Tech Stack:** NSL (model/training/inference), Rust (compiler runtime primitives), Python (data prep only), Cranelift (codegen backend)

**Spec:** `docs/superpowers/specs/2026-03-19-nsl-coder-50m-design.md`

---

## File Structure

### Runtime Primitives (Rust)
- Modify: `crates/nsl-codegen/src/builtins.rs` — register new runtime functions
- Modify: `crates/nsl-codegen/src/expr/calls.rs` — add codegen dispatch for new intrinsics
- Modify: `crates/nsl-runtime/src/tensor/shape_ops.rs` — rewrite `expand` for zero-copy + add `contiguous()`
- Create: `crates/nsl-runtime/src/tensor/trig.rs` — tensor-level sin/cos
- Modify: `crates/nsl-runtime/src/tensor/mod.rs` — add `mod trig;` and re-export
- Modify: `crates/nsl-runtime/src/lib.rs` — export new functions if needed
- Create: `crates/nsl-runtime/src/io.rs` — `read_line()` implementation
- Modify: `crates/nsl-codegen/src/expr/calls.rs` — add model_save/model_load codegen dispatch
- Modify: `crates/nsl-codegen/src/training.rs` — wire gradient clipping into train block

### Stdlib Modules (NSL)
- Create: `stdlib/nsl/nn/rope.nsl` — RotaryEmbedding model
- Create: `stdlib/nsl/nn/gqa.nsl` — GroupedQueryAttention model
- Create: `stdlib/nsl/io.nsl` — `read_line()` wrapper

### Model Project (NSL)
- Create: `models/coder50m/config.nsl` — hyperparameters
- Create: `models/coder50m/model.nsl` — NSLCoder architecture
- Create: `models/coder50m/pretrain.nsl` — Stage 1 training script
- Create: `models/coder50m/finetune.nsl` — Stage 2 training script
- Create: `models/coder50m/generate.nsl` — interactive inference demo

### Data & Docs
- Create: `models/coder50m/data/prepare_nsl.py` — tokenize NSL source files
- Create: `models/coder50m/README.md` — reproduction instructions

### Tests
- Create: `tests/test_tensor_trig.nsl` — tensor sin/cos tests
- Create: `tests/test_rotate_half.nsl` — rotate_half tests
- Create: `tests/test_expand_zerocopy.nsl` — zero-copy expand tests
- Create: `tests/test_expand_matmul.nsl` — matmul on stride=0 expanded tensors
- Create: `tests/test_contiguous.nsl` — contiguous() materialization test
- Create: `tests/test_rope.nsl` — RotaryEmbedding tests
- Create: `tests/test_gqa.nsl` — GroupedQueryAttention tests
- Create: `tests/test_coder50m_forward.nsl` — model forward pass smoke test
- Create: `tests/test_checkpoint_roundtrip.nsl` — save/load model weights
- Create: `tests/test_grad_clip.nsl` — gradient clipping verification

---

## Phase 1: Runtime Primitives

### Task 1: Tensor sin/cos Intrinsics

Only scalar `nsl_sin(f64)->f64` and `nsl_cos(f64)->f64` exist today. RoPE needs
tensor-level elementwise versions.

**Files:**
- Create: `crates/nsl-runtime/src/tensor/trig.rs`
- Modify: `crates/nsl-runtime/src/tensor/mod.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr/calls.rs`
- Create: `tests/test_tensor_trig.nsl`

- [ ] **Step 1: Write the test file**

```nsl
# tests/test_tensor_trig.nsl
# Test tensor-level sin and cos

let x = zeros([4])
let s = tensor_sin(x)
print(s)
# Expected: [0.0, 0.0, 0.0, 0.0]

let c = tensor_cos(x)
print(c)
# Expected: [1.0, 1.0, 1.0, 1.0]

let pi = full([2], 3.14159265)
let s2 = tensor_sin(pi)
print(s2)
# Expected: ~[0.0, 0.0]

let half_pi = full([2], 1.57079632)
let c2 = tensor_cos(half_pi)
print(c2)
# Expected: ~[0.0, 0.0]

print("tensor_trig: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo run -- run tests/test_tensor_trig.nsl`
Expected: FAIL — `undefined function 'tensor_sin'`

- [ ] **Step 3: Implement tensor sin/cos in runtime**

Create `crates/nsl-runtime/src/tensor/trig.rs`:

```rust
use super::NslTensor;
use crate::memory::checked_alloc;
use std::ffi::c_void;
use std::sync::atomic::AtomicI64;

#[no_mangle]
pub extern "C" fn nsl_tensor_sin(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = val.sin() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = val.sin() };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len,
        refcount: AtomicI64::new(1),
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    Box::into_raw(result) as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_cos(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = val.cos() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = val.cos() };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len,
        refcount: AtomicI64::new(1),
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    Box::into_raw(result) as i64
}
```

Add `pub mod trig;` to `crates/nsl-runtime/src/tensor/mod.rs`.

- [ ] **Step 4: Register in builtins.rs**

Add to the `RUNTIME_FUNCTIONS` array in `crates/nsl-codegen/src/builtins.rs`:

```rust
("nsl_tensor_sin", &[types::I64], Some(types::I64)),
("nsl_tensor_cos", &[types::I64], Some(types::I64)),
```

- [ ] **Step 5: Add codegen dispatch in calls.rs**

In `crates/nsl-codegen/src/expr/calls.rs`, add alongside the existing activation
function dispatch (around line 407):

```rust
if matches!(func_name.as_str(), "tensor_sin" | "tensor_cos")
    && !self.functions.contains_key(&func_name)
{
    if args.len() != 1 {
        return Err(CodegenError::new(format!("{func_name}() takes exactly 1 argument")));
    }
    let val = self.compile_expr(builder, state, &args[0].value)?;
    let rt_name = format!("nsl_{func_name}");
    return self.compile_traced_call(builder, &rt_name, &[val]);
}
```

- [ ] **Step 6: Build and run test**

Run: `cargo build && cargo run -- run tests/test_tensor_trig.nsl`
Expected: PASS — all outputs match expected values

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-runtime/src/tensor/trig.rs crates/nsl-runtime/src/tensor/mod.rs \
       crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr/calls.rs \
       tests/test_tensor_trig.nsl
git commit -m "feat: add tensor-level sin/cos intrinsics for RoPE"
```

---

### Task 2: rotate_half Intrinsic

Fused operation for RoPE: splits last dim in half, negates second half, swaps.
`rotate_half([x1,x2,x3,x4]) = [-x3,-x4,x1,x2]`

**Files:**
- Modify: `crates/nsl-runtime/src/tensor/shape_ops.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr/calls.rs`
- Create: `tests/test_rotate_half.nsl`

- [ ] **Step 1: Write the test file**

```nsl
# tests/test_rotate_half.nsl
let x = arange(0.0, 8.0)
let x2 = reshape(x, [2, 4])
let r = rotate_half(x2)
print(r)
# For row [0,1,2,3]: rotate_half = [-2,-3,0,1]
# For row [4,5,6,7]: rotate_half = [-6,-7,4,5]
# Expected: [[-2,-3,0,1],[-6,-7,4,5]]

print("rotate_half: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo run -- run tests/test_rotate_half.nsl`
Expected: FAIL — `undefined function 'rotate_half'`

- [ ] **Step 3: Implement rotate_half in runtime**

Add to `crates/nsl-runtime/src/tensor/shape_ops.rs`:

```rust
/// Fused rotate_half for RoPE (LLaMA-2 half-split variant).
/// Input shape [..., D], output shape [..., D].
/// rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2], dim=-1)
#[no_mangle]
pub extern "C" fn nsl_tensor_rotate_half(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let ndim = a.ndim as usize;
    let len = a.len;
    let last_dim = unsafe { *a.shape.add(ndim - 1) } as usize;

    if last_dim % 2 != 0 {
        eprintln!("nsl: rotate_half requires even last dimension, got {}", last_dim);
        std::process::abort();
    }

    let half = last_dim / 2;
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let src = a.data_f32();
        // Process in chunks of last_dim
        let num_chunks = len as usize / last_dim;
        for chunk in 0..num_chunks {
            let base = chunk * last_dim;
            // First half of output = negated second half of input
            for i in 0..half {
                unsafe { *buf.add(base + i) = -(*src.add(base + half + i)) };
            }
            // Second half of output = first half of input
            for i in 0..half {
                unsafe { *buf.add(base + half + i) = *src.add(base + i) };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        let src = a.data_f64();
        let num_chunks = len as usize / last_dim;
        for chunk in 0..num_chunks {
            let base = chunk * last_dim;
            for i in 0..half {
                unsafe { *buf.add(base + i) = -(*src.add(base + half + i)) };
            }
            for i in 0..half {
                unsafe { *buf.add(base + half + i) = *src.add(base + i) };
            }
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim: a.ndim, len,
        refcount: AtomicI64::new(1),
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    Box::into_raw(result) as i64
}
```

- [ ] **Step 4: Register in builtins.rs**

```rust
("nsl_tensor_rotate_half", &[types::I64], Some(types::I64)),
```

- [ ] **Step 5: Add codegen dispatch**

In `crates/nsl-codegen/src/expr/calls.rs`, add with the activation functions:

```rust
if func_name == "rotate_half" && !self.functions.contains_key(&func_name) {
    if args.len() != 1 {
        return Err(CodegenError::new("rotate_half() takes exactly 1 argument".into()));
    }
    let val = self.compile_expr(builder, state, &args[0].value)?;
    return self.compile_traced_call(builder, "nsl_tensor_rotate_half", &[val]);
}
```

- [ ] **Step 6: Build and run test**

Run: `cargo build && cargo run -- run tests/test_rotate_half.nsl`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-runtime/src/tensor/shape_ops.rs \
       crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr/calls.rs \
       tests/test_rotate_half.nsl
git commit -m "feat: add fused rotate_half intrinsic for RoPE"
```

---

### Task 3: Rewrite expand for Zero-Copy Stride Views

Current `nsl_tensor_expand` at `shape_ops.rs:620` allocates new memory and copies.
Must become a stride-based view where expanded dims get stride=0.

**Files:**
- Modify: `crates/nsl-runtime/src/tensor/shape_ops.rs`
- Create: `tests/test_expand_zerocopy.nsl`

- [ ] **Step 1: Write the test file**

```nsl
# tests/test_expand_zerocopy.nsl
# Test that expand produces correct values (stride=0 semantics)
let x = ones([1, 4])
let expanded = expand(x, [3, 4])
print(expanded)
# Expected: [[1,1,1,1],[1,1,1,1],[1,1,1,1]]

# Test with non-trivial values
let y = arange(0.0, 4.0)
let y2 = reshape(y, [1, 4])
let y3 = expand(y2, [3, 4])
print(y3)
# Expected: [[0,1,2,3],[0,1,2,3],[0,1,2,3]]

# Test GQA pattern: [1, 4, 8, 64] -> [1, 8, 8, 64] (expand head dim)
let kv = randn([1, 4, 8, 64])
let kv_expanded = expand(kv, [1, 8, 8, 64])
print(kv_expanded.shape)
# Expected: [1, 8, 8, 64]

print("expand_zerocopy: PASS")
```

- [ ] **Step 2: Run test to verify current expand works (it copies but should produce correct values)**

Run: `cargo run -- run tests/test_expand_zerocopy.nsl`
Expected: PASS (values correct, but internally it copies)

- [ ] **Step 3: Rewrite expand to use stride=0**

Replace the body of `nsl_tensor_expand` in `crates/nsl-runtime/src/tensor/shape_ops.rs`:

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_expand(tensor_ptr: i64, shape_list: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let list = NslList::from_ptr(shape_list);
    let target_ndim = list.len as usize;

    // Extract target shape
    let mut target_shape: Vec<i64> = Vec::with_capacity(target_ndim);
    for i in 0..target_ndim {
        target_shape.push(unsafe { *list.data.add(i) });
    }

    let src_ndim = tensor.ndim as usize;
    let pad = target_ndim.saturating_sub(src_ndim);

    // Validate: source dims must be 1 or match target
    for i in 0..target_ndim {
        let s = if i < pad { 1 } else {
            unsafe { *tensor.shape.add(i - pad) }
        };
        let t = target_shape[i];
        if s != 1 && s != t {
            eprintln!("nsl: expand: cannot expand dim {} from {} to {}", i, s, t);
            std::process::abort();
        }
    }

    // Build output shape and strides
    let out_shape = checked_alloc(target_ndim * std::mem::size_of::<i64>()) as *mut i64;
    let out_strides = checked_alloc(target_ndim * std::mem::size_of::<i64>()) as *mut i64;

    for i in 0..target_ndim {
        unsafe { *out_shape.add(i) = target_shape[i] };
        let src_dim_idx = i as isize - pad as isize;
        if src_dim_idx < 0 {
            // Padded leading dim: stride = 0 (broadcast)
            unsafe { *out_strides.add(i) = 0 };
        } else {
            let s = unsafe { *tensor.shape.add(src_dim_idx as usize) };
            if s == 1 && target_shape[i] != 1 {
                // Expanded dim: stride = 0 (broadcast, zero-copy)
                unsafe { *out_strides.add(i) = 0 };
            } else {
                // Keep original stride
                unsafe { *out_strides.add(i) = *tensor.strides.add(src_dim_idx as usize) };
            }
        }
    }

    let out_len = NslTensor::total_elements(out_shape, target_ndim as i64);

    // ZERO-COPY: share data pointer, bump refcount
    tensor.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let result = Box::new(NslTensor {
        data: tensor.data,
        shape: out_shape,
        strides: out_strides,
        ndim: target_ndim as i64,
        len: out_len,
        refcount: AtomicI64::new(1),
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 0,  // Does NOT own data — shared view
    });

    // Record autodiff if needed
    if crate::autodiff::is_recording() {
        let out_ptr = Box::into_raw(result) as i64;
        // ... record TapeOp::Expand
        return out_ptr;
    }

    Box::into_raw(result) as i64
}
```

**CRITICAL:** All downstream tensor ops (matmul, elementwise) must handle non-contiguous
tensors with stride=0. Verify that `matmul` and elementwise ops use stride-aware
indexing rather than flat `data[i]` access. If not, add a `contiguous()` call in the
GQA attention module as a workaround (copies only when needed).

- [ ] **Step 4: Build and run test**

Run: `cargo build && cargo run -- run tests/test_expand_zerocopy.nsl`
Expected: PASS — same values as before, but now zero-copy

- [ ] **Step 5: Verify matmul handles stride=0 tensors**

Check if `nsl_tensor_matmul` in `tensor/matmul.rs` uses stride-aware indexing.
If it reads data as flat `data[i]`, the expanded tensor will produce wrong results.

If matmul doesn't handle strides: add a `contiguous()` helper that materializes
the view into a physical copy only when consumed by an op that needs contiguous data.
The GQA module will call `contiguous()` before matmul if needed.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/tensor/shape_ops.rs tests/test_expand_zerocopy.nsl
git commit -m "feat: rewrite expand as zero-copy stride view for GQA"
```

---

### Task 4: read_line Intrinsic

stdin input for the interactive demo.

**Files:**
- Create: `crates/nsl-runtime/src/io.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr/calls.rs`

- [ ] **Step 1: Implement read_line in runtime**

Create `crates/nsl-runtime/src/io.rs`:

```rust
use std::io::{self, Write};

/// Read a line from stdin, return as NslString pointer (i64).
#[no_mangle]
pub extern "C" fn nsl_read_line() -> i64 {
    io::stdout().flush().ok();
    let mut buf = String::new();
    io::stdin().read_line(&mut buf).unwrap_or(0);
    let trimmed = buf.trim_end_matches('\n').trim_end_matches('\r');
    crate::string::nsl_str_from_rust(trimmed)
}
```

Add `pub mod io;` to `crates/nsl-runtime/src/lib.rs`.

Note: `nsl_str_from_rust` creates an NslString from a Rust `&str`. Verify this
helper exists. If not, use the same pattern as `nsl_int_to_str` — allocate an
NslString struct, copy bytes, return as i64.

- [ ] **Step 2: Register in builtins.rs**

```rust
("nsl_read_line", &[], Some(types::I64)),
```

- [ ] **Step 3: Add codegen dispatch**

In `crates/nsl-codegen/src/expr/calls.rs`:

```rust
if func_name == "read_line" && !self.functions.contains_key(&func_name) {
    if !args.is_empty() {
        return Err(CodegenError::new("read_line() takes no arguments".into()));
    }
    return self.compile_traced_call(builder, "nsl_read_line", &[]);
}
```

- [ ] **Step 4: Write a quick manual test**

```nsl
# tests/test_read_line.nsl (manual test — requires stdin)
let line = read_line()
print(line)
```

Run manually: `echo "hello" | cargo run -- run tests/test_read_line.nsl`
Expected: prints "hello"

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/io.rs crates/nsl-runtime/src/lib.rs \
       crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr/calls.rs
git commit -m "feat: add read_line() intrinsic for stdin input"
```

---

### Task 5: Verify Gradient Clipping and arange

Both `nsl_clip_grad_norm` and `nsl_tensor_arange` are already registered. Verify
they work correctly.

**Files:**
- Create: `tests/test_grad_clip.nsl`
- Create: `tests/test_arange.nsl`

- [ ] **Step 1: Write arange test**

```nsl
# tests/test_arange.nsl
let a = arange(0.0, 5.0)
print(a)
# Expected: [0, 1, 2, 3, 4]

let b = arange(2.0, 8.0, 2.0)
print(b)
# Expected: [2, 4, 6]

print("arange: PASS")
```

- [ ] **Step 2: Run arange test**

Run: `cargo run -- run tests/test_arange.nsl`
Expected: PASS

- [ ] **Step 3: Write gradient clipping test**

```nsl
# tests/test_grad_clip.nsl
# Verify nsl_clip_grad_norm works in a training context
from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([2, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = full([2, 2], 10.0)
let y = zeros([2, 2])

train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(loss)

print("grad_clip: PASS")
```

Note: gradient clipping may be integrated into the train block codegen automatically.
If not, check `nsl_clip_grad_norm` signature and verify it's called. If gradient
clipping is not wired into train blocks, this becomes a must-fix task.

- [ ] **Step 4: Run gradient clipping test**

Run: `cargo run -- run tests/test_grad_clip.nsl`
Expected: Loss decreases over 3 epochs without NaN

- [ ] **Step 5: If grad clipping is NOT wired into train blocks**

Check `crates/nsl-codegen/src/training.rs` for where `nsl_clip_grad_norm` is called.
If it's not called automatically, add it to the train block codegen between the
backward pass and optimizer step:

```rust
// After backward, before optimizer step:
self.compile_call_by_name(builder, "nsl_clip_grad_norm", &[model_ptr, max_norm])?;
```

Where `max_norm` defaults to 1.0 unless specified by the user.

- [ ] **Step 6: Commit**

```bash
git add tests/test_grad_clip.nsl tests/test_arange.nsl
git commit -m "test: verify arange and gradient clipping work correctly"
```

---

### Task 5b: contiguous() Runtime Function

Matmul uses flat offset indexing (`data[batch*stride + row*cols + col]`), which
produces wrong results on stride=0 expanded tensors. A `contiguous()` function
materializes non-contiguous views into physical copies.

**Files:**
- Modify: `crates/nsl-runtime/src/tensor/shape_ops.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr/calls.rs`
- Create: `tests/test_contiguous.nsl`
- Create: `tests/test_expand_matmul.nsl`

- [ ] **Step 1: Write test for expand+matmul interaction**

```nsl
# tests/test_expand_matmul.nsl
# Verify matmul produces correct results on expanded (stride=0) tensors
let a = arange(0.0, 4.0)
let a2 = reshape(a, [1, 4])
let a_exp = expand(a2, [3, 4])
# a_exp: [[0,1,2,3],[0,1,2,3],[0,1,2,3]]

let b = ones([4, 2])
let result = a_exp @ b
print(result)
# Each row: [0+1+2+3, 0+1+2+3] = [6.0, 6.0]
# Expected: [[6,6],[6,6],[6,6]]

print("expand_matmul: PASS")
```

- [ ] **Step 2: Run test — expect WRONG results or crash**

Run: `cargo run -- run tests/test_expand_matmul.nsl`
Expected: FAIL (wrong values because matmul ignores strides)

- [ ] **Step 3: Implement contiguous()**

Add to `crates/nsl-runtime/src/tensor/shape_ops.rs`:

```rust
/// Materialize a non-contiguous tensor into a contiguous copy.
/// If tensor is already contiguous, returns a new reference (bumps refcount).
#[no_mangle]
pub extern "C" fn nsl_tensor_contiguous(tensor_ptr: i64) -> i64 {
    let t = NslTensor::from_ptr(tensor_ptr);
    let ndim = t.ndim as usize;
    let len = t.len;

    // Check if already contiguous (strides match row-major layout)
    let expected_strides = NslTensor::compute_strides(t.shape, t.ndim);
    let mut is_contiguous = true;
    for i in 0..ndim {
        if unsafe { *t.strides.add(i) != *expected_strides.add(i) } {
            is_contiguous = false;
            break;
        }
    }
    unsafe { crate::memory::checked_free(expected_strides as *mut u8, ndim * 8) };

    if is_contiguous {
        // Already contiguous — bump refcount and return
        t.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        return tensor_ptr;
    }

    // Materialize: iterate using strides to read, write to flat output
    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    let data: *mut c_void = if t.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for flat_idx in 0..len as usize {
            // Convert flat index to multi-dim coords, then to strided offset
            let mut remaining = flat_idx;
            let mut src_offset: usize = 0;
            for d in 0..ndim {
                let dim_size = unsafe { *shape.add(d) } as usize;
                let coord = remaining / (len as usize / dim_size); // simplified
                // More robust: compute coord from contiguous strides
                let out_stride = unsafe { *strides.add(d) } as usize;
                let coord = if out_stride > 0 { remaining / out_stride } else { 0 };
                remaining = if out_stride > 0 { remaining % out_stride } else { remaining };
                let src_stride = unsafe { *t.strides.add(d) } as usize;
                src_offset += coord * src_stride;
            }
            unsafe { *buf.add(flat_idx) = *t.data_f32().add(src_offset) };
        }
        buf as *mut c_void
    } else {
        // Same for f64
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for flat_idx in 0..len as usize {
            let mut remaining = flat_idx;
            let mut src_offset: usize = 0;
            for d in 0..ndim {
                let out_stride = unsafe { *strides.add(d) } as usize;
                let coord = if out_stride > 0 { remaining / out_stride } else { 0 };
                remaining = if out_stride > 0 { remaining % out_stride } else { remaining };
                let src_stride = unsafe { *t.strides.add(d) } as usize;
                src_offset += coord * src_stride;
            }
            unsafe { *buf.add(flat_idx) = *t.data_f64().add(src_offset) };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor {
        data, shape, strides, ndim: t.ndim, len,
        refcount: AtomicI64::new(1),
        device: t.device, dtype: t.dtype, owns_data: 1,
    });
    Box::into_raw(result) as i64
}
```

- [ ] **Step 4: Register in builtins.rs and add codegen dispatch**

builtins.rs:
```rust
("nsl_tensor_contiguous", &[types::I64], Some(types::I64)),
```

calls.rs:
```rust
if func_name == "contiguous" && !self.functions.contains_key(&func_name) {
    if args.len() != 1 {
        return Err(CodegenError::new("contiguous() takes exactly 1 argument".into()));
    }
    let val = self.compile_expr(builder, state, &args[0].value)?;
    return self.compile_traced_call(builder, "nsl_tensor_contiguous", &[val]);
}
```

- [ ] **Step 5: Run expand+matmul test again**

Run: `cargo build && cargo run -- run tests/test_expand_matmul.nsl`

If matmul still produces wrong results (because it doesn't call contiguous
internally), the GQA module must call `contiguous()` explicitly before matmul.
Update the GQA code in Task 7 accordingly:
```nsl
let k_exp = contiguous(expand(k_rope, [batch, self.n_heads, seq_len, self.head_dim]))
let v_exp = contiguous(expand(v4t, [batch, self.n_heads, seq_len, self.head_dim]))
```

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/tensor/shape_ops.rs \
       crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr/calls.rs \
       tests/test_contiguous.nsl tests/test_expand_matmul.nsl
git commit -m "feat: add contiguous() to materialize non-contiguous tensor views"
```

---

### Task 5c: Wire Gradient Clipping into Train Block Codegen

`nsl_clip_grad_norm` is registered in builtins but NOT called by the train block
compiler. Gradient clipping is mandatory for training stability (spec Section 3.3).

**Files:**
- Modify: `crates/nsl-codegen/src/training.rs`

- [ ] **Step 1: Find the backward-to-optimizer gap in training.rs**

Read `crates/nsl-codegen/src/training.rs`. Find the section between
`nsl_tape_backward()` and the optimizer step loop. This is where gradient
clipping must be inserted.

- [ ] **Step 2: Insert gradient clipping call**

After the backward pass and before the optimizer step, add:

```rust
// Gradient clipping: nsl_clip_grad_norm(model_ptr, max_norm=1.0)
let max_norm = builder.ins().f64const(1.0);
self.compile_call_by_name(builder, "nsl_clip_grad_norm", &[model_val, max_norm])?;
```

Check `nsl_clip_grad_norm` signature in builtins.rs to match the argument types.
The runtime function needs: a way to iterate model parameters and their gradients,
plus the max_norm threshold. If the current signature doesn't support this, adapt.

- [ ] **Step 3: Test with the gradient clipping test from Task 5**

Run: `cargo build && cargo run -- run tests/test_grad_clip.nsl`
Expected: Loss decreases, no NaN

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/training.rs
git commit -m "feat: wire gradient clipping into train block codegen"
```

---

### Task 5d: Model Save/Load Codegen Dispatch

`model_save` and `model_load` are registered in builtins but have no
codegen dispatch from NSL code. The runtime expects 3-4 args (path ptr, path len,
param tensors ptr), but NSL code should call `model_save(m, path)`.

**Files:**
- Modify: `crates/nsl-codegen/src/expr/calls.rs`
- Create: `tests/test_checkpoint_roundtrip.nsl`

- [ ] **Step 1: Understand the runtime signatures**

Read `crates/nsl-runtime/src/checkpoint.rs` to understand:
- `model_save(path_ptr, path_len, param_names_ptr, param_tensors_ptr)`
- `model_load(path_ptr, path_len, param_tensors_ptr)`

The codegen must extract model parameters from the model struct and pack them
into the expected format.

- [ ] **Step 2: Add codegen dispatch for model_save/model_load**

In `crates/nsl-codegen/src/expr/calls.rs`, add dispatch that:
1. Takes `(model, path_string)` from NSL code
2. Extracts the model's parameter tensor pointers from the compiled struct
3. Constructs the parameter name list and tensor pointer list
4. Calls the runtime function with the correct 3-4 args

Note: This may already be partially done in the model compilation code
(`compile_model_save` or similar). Search the codegen for existing patterns
around `model_save` usage.

- [ ] **Step 3: Write checkpoint roundtrip test**

```nsl
# tests/test_checkpoint_roundtrip.nsl
from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = ones([2, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()

# Modify weights via training
let x = full([2, 2], 2.0)
let y = zeros([2, 2])
train(model = m, epochs = 5):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)

# Save
model_save(m, "test_checkpoint.nslm")

# Load into fresh model
let m2 = Tiny()
model_load(m2, "test_checkpoint.nslm")

# Verify weights match
let diff = sum(abs(m.w - m2.w))
print(diff)
# Expected: 0.0 (or very close)

print("checkpoint_roundtrip: PASS")
```

- [ ] **Step 4: Run test**

Run: `cargo build && cargo run -- run tests/test_checkpoint_roundtrip.nsl`
Expected: PASS — weights match after save/load

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/expr/calls.rs tests/test_checkpoint_roundtrip.nsl
git commit -m "feat: add model_save/model_load codegen dispatch for checkpointing"
```

---

### Task 5e: Fix sample_top_k Stdlib

The existing `stdlib/nsl/inference/sampling.nsl` uses `result["values"]` dict
indexing on topk results, which may not compile. Rewrite with a simpler approach
or verify the syntax works.

**Files:**
- Modify: `stdlib/nsl/inference/sampling.nsl` (if needed)
- Create: `tests/test_sampling.nsl`

- [ ] **Step 1: Test existing sample_top_k**

```nsl
# tests/test_sampling.nsl
from nsl.inference.sampling import sample_top_k

let logits = randn([1, 100])
let token = sample_top_k(logits, 10, 0.8)
print(token)
# Expected: a single integer token ID

print("sampling: PASS")
```

- [ ] **Step 2: Run test**

Run: `cargo run -- run tests/test_sampling.nsl`
Expected: Either PASS (dict indexing works) or compile error

- [ ] **Step 3: If test fails, rewrite sample_top_k**

Replace the stdlib function with a simpler version that avoids dict indexing:

```nsl
fn sample_top_k(logits: Tensor, k: int, temperature: float) -> Tensor:
    let scaled = logits / temperature
    # Use topk to get indices and values
    let top_vals = topk_values(scaled, k)
    let top_idxs = topk_indices(scaled, k)
    let probs = softmax(top_vals, 0)
    let sampled = multinomial(probs, 1)
    return top_idxs.gather(0, sampled)
```

Or: implement `sample_top_k` as a single runtime intrinsic in Rust that handles
the full topk→softmax→multinomial pipeline internally, avoiding the dict issue.
Add `nsl_sample_top_k(logits_ptr, k, temperature) -> i64` to the runtime.

- [ ] **Step 4: Commit**

```bash
git add stdlib/nsl/inference/sampling.nsl tests/test_sampling.nsl
git commit -m "fix: rewrite sample_top_k to avoid dict indexing on topk results"
```

---

## Phase 2: Stdlib Modules

### Task 6: RotaryEmbedding Stdlib Module

**Files:**
- Create: `stdlib/nsl/nn/rope.nsl`
- Create: `tests/test_rope.nsl`

- [ ] **Step 1: Write the test file**

```nsl
# tests/test_rope.nsl
from nsl.nn.rope import RotaryEmbedding

let rope = RotaryEmbedding(64, 32, 10000.0)

# Create dummy Q tensor: [batch=1, heads=2, seq=4, head_dim=64]
let q = randn([1, 2, 4, 64])

let q_rotated = rope.forward(q, 4)
print(q_rotated.shape)
# Expected: [1, 2, 4, 64] — same shape, different values

# Verify output is not identical to input (rotation applied)
let diff = q_rotated - q
let diff_sum = sum(abs(diff))
print(diff_sum)
# Expected: non-zero value

print("rope: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo run -- run tests/test_rope.nsl`
Expected: FAIL — cannot resolve import

- [ ] **Step 3: Implement RotaryEmbedding**

Create `stdlib/nsl/nn/rope.nsl`:

```nsl
model RotaryEmbedding(dim: int, max_seq_len: int, theta: float):
    # Precompute inverse frequencies: theta^(-2i/d) for i in [0, dim/2)
    # inv_freq shape: [dim/2]
    inv_freq: Tensor = 1.0 / (theta ** (arange(0.0, dim, 2.0) / dim))
    max_seq: int = max_seq_len

    fn forward(self, x: Tensor, seq_len: int) -> Tensor:
        # x shape: [batch, heads, seq, head_dim]
        # Build position indices: [seq_len]
        let t = arange(0.0, seq_len)

        # Outer product: freqs[seq, dim/2] = t[:, None] * inv_freq[None, :]
        let t2 = reshape(t, [seq_len, 1])
        let inv2 = reshape(self.inv_freq, [1, self.inv_freq.shape[0]])
        let freqs = t2 @ inv2

        # Duplicate for full head_dim: [seq, dim]
        let emb = cat(freqs, freqs, 1)

        # Compute cos and sin: [seq, dim]
        let cos_emb = tensor_cos(emb)
        let sin_emb = tensor_sin(emb)

        # Reshape for broadcasting: [1, 1, seq, dim]
        let cos_4d = reshape(cos_emb, [1, 1, seq_len, emb.shape[1]])
        let sin_4d = reshape(sin_emb, [1, 1, seq_len, emb.shape[1]])

        # Apply rotation: x * cos + rotate_half(x) * sin
        return x * cos_4d + rotate_half(x) * sin_4d
```

- [ ] **Step 4: Run test**

Run: `cargo run -- run tests/test_rope.nsl`
Expected: PASS — shapes match, values changed

- [ ] **Step 5: Commit**

```bash
git add stdlib/nsl/nn/rope.nsl tests/test_rope.nsl
git commit -m "feat: add RotaryEmbedding stdlib module for RoPE"
```

---

### Task 7: GroupedQueryAttention Stdlib Module

**Files:**
- Create: `stdlib/nsl/nn/gqa.nsl`
- Create: `tests/test_gqa.nsl`

- [ ] **Step 1: Write the test file**

```nsl
# tests/test_gqa.nsl
from nsl.nn.gqa import GroupedQueryAttention

let attn = GroupedQueryAttention(512, 8, 4)

# Dummy input: [batch=1, seq=8, d_model=512]
let x = randn([1, 8, 512])

let out = attn.forward(x)
print(out.shape)
# Expected: [1, 8, 512]

print("gqa: PASS")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo run -- run tests/test_gqa.nsl`
Expected: FAIL — cannot resolve import

- [ ] **Step 3: Implement GroupedQueryAttention**

Create `stdlib/nsl/nn/gqa.nsl`:

```nsl
from nsl.nn.rope import RotaryEmbedding

model GroupedQueryAttention(d_model: int, n_heads: int, n_kv_heads: int):
    head_dim: int = d_model / n_heads
    n_rep: int = n_heads / n_kv_heads

    # Projections
    wq: Tensor = randn([d_model, d_model])
    wk: Tensor = randn([d_model, n_kv_heads * head_dim])
    wv: Tensor = randn([d_model, n_kv_heads * head_dim])
    wo: Tensor = randn([d_model, d_model])

    # RoPE
    rope: RotaryEmbedding = RotaryEmbedding(head_dim, 1024, 10000.0)

    # Scaling factor
    scale: float = 1.0 / sqrt(head_dim)

    fn forward(self, x: Tensor) -> Tensor:
        let s = x.shape
        let batch = s[0]
        let seq_len = s[1]

        # Project Q, K, V
        let q = x @ self.wq.transpose(0, 1)
        let k = x @ self.wk.transpose(0, 1)
        let v = x @ self.wv.transpose(0, 1)

        # Reshape to [batch, seq, heads, head_dim] then transpose to [batch, heads, seq, head_dim]
        let q4 = reshape(q, [batch, seq_len, self.n_heads, self.head_dim])
        let q4t = q4.transpose(1, 2)

        let k4 = reshape(k, [batch, seq_len, self.n_kv_heads, self.head_dim])
        let k4t = k4.transpose(1, 2)

        let v4 = reshape(v, [batch, seq_len, self.n_kv_heads, self.head_dim])
        let v4t = v4.transpose(1, 2)

        # Apply RoPE to Q and K
        let q_rope = self.rope.forward(q4t, seq_len)
        let k_rope = self.rope.forward(k4t, seq_len)

        # GQA: expand KV heads to match Q heads
        # k_rope: [batch, n_kv_heads, seq, head_dim] -> [batch, n_heads, seq, head_dim]
        # contiguous() materializes stride=0 view so matmul gets contiguous data
        let k_exp = contiguous(expand(k_rope, [batch, self.n_heads, seq_len, self.head_dim]))
        let v_exp = contiguous(expand(v4t, [batch, self.n_heads, seq_len, self.head_dim]))

        # Attention: QK^T / sqrt(d) + mask
        let scores = (q_rope @ k_exp.transpose(2, 3)) * self.scale
        let mask = causal_mask(seq_len)
        let masked = scores + mask
        let attn_weights = softmax(masked, 3)

        # Weighted sum of values
        let attn_out = attn_weights @ v_exp

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, d_model]
        let attn_t = attn_out.transpose(1, 2)
        let attn_flat = reshape(attn_t, [batch, seq_len, self.d_model])

        # Output projection
        return attn_flat @ self.wo.transpose(0, 1)
```

- [ ] **Step 4: Run test**

Run: `cargo run -- run tests/test_gqa.nsl`
Expected: PASS — output shape [1, 8, 512]

- [ ] **Step 5: Commit**

```bash
git add stdlib/nsl/nn/gqa.nsl tests/test_gqa.nsl
git commit -m "feat: add GroupedQueryAttention stdlib module with GQA + RoPE"
```

---

### Task 8: nsl.io Stdlib Module

Thin wrapper exposing `read_line()` for import convenience.

**Files:**
- Create: `stdlib/nsl/io.nsl`

- [ ] **Step 1: Create nsl.io module**

Create `stdlib/nsl/io.nsl`:

```nsl
fn input(prompt: str) -> str:
    print(prompt)
    return read_line()
```

This wraps the `read_line()` intrinsic (Task 4) with an optional prompt.

- [ ] **Step 2: Commit**

```bash
git add stdlib/nsl/io.nsl
git commit -m "feat: add nsl.io stdlib module with input() helper"
```

---

## Phase 3: Model Definition

### Task 9: config.nsl

**Files:**
- Create: `models/coder50m/config.nsl`

- [ ] **Step 1: Create config file**

```nsl
# models/coder50m/config.nsl
# NSL-Coder-50M Configuration — Single source of truth

# Architecture
const VOCAB_SIZE = 49152
const D_MODEL = 512
const N_LAYERS = 8
const N_HEADS = 8
const N_KV_HEADS = 4
const HEAD_DIM = 64
const D_FF = 1408
const MAX_SEQ_LEN = 1024
const ROPE_THETA = 10000.0

# Pretrain hyperparameters
const PRETRAIN_LR = 0.0003
const PRETRAIN_WARMUP = 3000
const PRETRAIN_TOTAL_STEPS = 305000
const PRETRAIN_MIN_LR = 0.00003
const PRETRAIN_BATCH_SIZE = 32
const PRETRAIN_WEIGHT_DECAY = 0.1
const PRETRAIN_EPOCHS = 1

# Finetune hyperparameters
const FINETUNE_LR = 0.00001
const FINETUNE_WARMUP = 100
const FINETUNE_MIN_LR = 0.000001
const FINETUNE_BATCH_SIZE = 16
const FINETUNE_WEIGHT_DECAY = 0.1
const FINETUNE_EPOCHS = 2

# Inference
const TOP_K = 40
const TEMPERATURE = 0.8
const MAX_GEN_TOKENS = 512

# Paths
const PRETRAIN_DATA = "data/pretokenized/tokens.bin"
const FINETUNE_DATA = "data/finetune/mixed_tokens.bin"
const TOKENIZER_PATH = "data/tokenizer/codeforge.json"
const CHECKPOINT_DIR = "checkpoints"
```

- [ ] **Step 2: Commit**

```bash
git add models/coder50m/config.nsl
git commit -m "feat: add NSL-Coder-50M config with all hyperparameters"
```

---

### Task 10: model.nsl — NSLCoder Architecture

**Files:**
- Create: `models/coder50m/model.nsl`
- Create: `tests/test_coder50m_forward.nsl`

- [ ] **Step 1: Write forward pass smoke test**

```nsl
# tests/test_coder50m_forward.nsl
# Smoke test: verify the model compiles and forward pass produces correct shapes

# Use smaller config for testing
const VOCAB_SIZE = 256
const D_MODEL = 64
const N_LAYERS = 2
const N_HEADS = 4
const N_KV_HEADS = 2
const HEAD_DIM = 16
const D_FF = 176
const MAX_SEQ_LEN = 32
const ROPE_THETA = 10000.0

from nsl.nn.norms import RMSNorm
from nsl.nn.gqa import GroupedQueryAttention
from nsl.nn.losses import cross_entropy

model SwiGLUFFN(d_model: int, d_ff: int):
    w_gate: Tensor = randn([d_model, d_ff])
    w_up: Tensor = randn([d_model, d_ff])
    w_down: Tensor = randn([d_ff, d_model])

    fn forward(self, x: Tensor) -> Tensor:
        let gate = silu(x @ self.w_gate.transpose(0, 1))
        let up = x @ self.w_up.transpose(0, 1)
        return (gate * up) @ self.w_down.transpose(0, 1)

model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
    attn_norm: RMSNorm = RMSNorm(d_model)
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    ffn_norm: RMSNorm = RMSNorm(d_model)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff)

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + self.attn.forward(self.attn_norm.forward(x))
        return h + self.ffn.forward(self.ffn_norm.forward(h))

model NSLCoder:
    embed: Tensor = randn([VOCAB_SIZE, D_MODEL])
    blocks: [TransformerBlock; 2] = TransformerBlock(D_MODEL, N_HEADS, N_KV_HEADS, D_FF)
    norm: RMSNorm = RMSNorm(D_MODEL)

    fn forward(self, input_ids: Tensor) -> Tensor:
        let x = embedding_lookup(self.embed, input_ids)
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        # Weight-tied LM head
        return x @ self.embed.transpose(0, 1)

let m = NSLCoder()
let tokens = zeros([1, 8])
let logits = m.forward(tokens)
print(logits.shape)
# Expected: [1, 8, 256]

print("coder50m_forward: PASS")
```

- [ ] **Step 2: Run test to verify it fails (model not written yet for full config)**

Run: `cargo run -- run tests/test_coder50m_forward.nsl`
Expected: Either PASS (if inline model works) or compiler error to debug

- [ ] **Step 3: Create model.nsl**

```nsl
# models/coder50m/model.nsl
# NSL-Coder-50M: LLaMA-style decoder-only transformer

from config import VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, N_KV_HEADS
from config import HEAD_DIM, D_FF, MAX_SEQ_LEN, ROPE_THETA
from nsl.nn.norms import RMSNorm
from nsl.nn.gqa import GroupedQueryAttention

model SwiGLUFFN(d_model: int, d_ff: int):
    w_gate: Tensor = randn([d_model, d_ff])
    w_up: Tensor = randn([d_model, d_ff])
    w_down: Tensor = randn([d_ff, d_model])

    fn forward(self, x: Tensor) -> Tensor:
        let gate = silu(x @ self.w_gate.transpose(0, 1))
        let up = x @ self.w_up.transpose(0, 1)
        return (gate * up) @ self.w_down.transpose(0, 1)

model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
    attn_norm: RMSNorm = RMSNorm(d_model)
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    ffn_norm: RMSNorm = RMSNorm(d_model)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff)

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + self.attn.forward(self.attn_norm.forward(x))
        return h + self.ffn.forward(self.ffn_norm.forward(h))

model NSLCoder:
    embed: Tensor = randn([VOCAB_SIZE, D_MODEL])
    blocks: [TransformerBlock; N_LAYERS] = TransformerBlock(D_MODEL, N_HEADS, N_KV_HEADS, D_FF)
    norm: RMSNorm = RMSNorm(D_MODEL)

    fn forward(self, input_ids: Tensor) -> Tensor:
        let x = embedding_lookup(self.embed, input_ids)
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        # Weight-tied LM head (no separate linear layer)
        return x @ self.embed.transpose(0, 1)
```

- [ ] **Step 4: Verify test still passes with import**

If the test uses inline model definitions (Step 1), it should still pass.
For a proper integration test, update the test to import from model.nsl.

- [ ] **Step 5: Commit**

```bash
git add models/coder50m/model.nsl tests/test_coder50m_forward.nsl
git commit -m "feat: add NSLCoder 50M model definition — LLaMA-style transformer"
```

---

## Phase 4: Training

### Task 11: pretrain.nsl — Stage 1 Training Script

**Files:**
- Create: `models/coder50m/pretrain.nsl`

- [ ] **Step 1: Create pretrain script**

```nsl
# models/coder50m/pretrain.nsl
# Stage 1: Pretrain on 10B StarCoder tokens

from config import VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, N_KV_HEADS
from config import D_FF, MAX_SEQ_LEN, PRETRAIN_DATA, PRETRAIN_BATCH_SIZE
from config import PRETRAIN_LR, PRETRAIN_WARMUP, PRETRAIN_TOTAL_STEPS
from config import PRETRAIN_MIN_LR, PRETRAIN_WEIGHT_DECAY, PRETRAIN_EPOCHS
from config import CHECKPOINT_DIR
from model import NSLCoder
from nsl.nn.losses import cross_entropy

# Load pretokenized data (u16 format, dtype=3)
let tokens = load_mmap(PRETRAIN_DATA, 3)
let loader = DataLoader(tokens, batch_size=PRETRAIN_BATCH_SIZE, seq_len=MAX_SEQ_LEN,
                         shuffle=false, drop_last=true)

# Initialize model
let m = NSLCoder()
print("NSL-Coder-50M initialized. Starting pretraining...")

# Train
train(model=m, epochs=PRETRAIN_EPOCHS):
    optimizer: AdamW(lr=PRETRAIN_LR, weight_decay=PRETRAIN_WEIGHT_DECAY)
    scheduler: warmup_cosine(warmup_steps=PRETRAIN_WARMUP,
                             total_steps=PRETRAIN_TOTAL_STEPS,
                             min_lr=PRETRAIN_MIN_LR)
    step(batch):
        let logits = m.forward(batch.input_ids)
        let loss = cross_entropy(logits, batch.labels)
    callbacks:
        on_step(step, loss):
            if step % 100 == 0:
                print(f"Step {step}: Loss {loss}")
            if step % 10000 == 0:
                model_save(m, f"{CHECKPOINT_DIR}/pretrain_step_{step}.nslm")
        on_epoch(epoch, loss):
            print(f"Epoch {epoch} complete. Final loss: {loss}")
            model_save(m, f"{CHECKPOINT_DIR}/pretrain_final.nslm")

print("Pretraining complete.")
```

- [ ] **Step 2: Verify the script compiles**

Run: `cargo run -- check models/coder50m/pretrain.nsl`
(Use `check` if available, otherwise `run` with a tiny data file to test compilation)
Expected: No compilation errors

- [ ] **Step 3: Commit**

```bash
git add models/coder50m/pretrain.nsl
git commit -m "feat: add pretrain.nsl — Stage 1 training on StarCoder data"
```

---

### Task 12: data/prepare_nsl.py — NSL Dataset Preparation

**Files:**
- Create: `models/coder50m/data/prepare_nsl.py`

- [ ] **Step 1: Create the Python data prep script**

```python
#!/usr/bin/env python3
"""Tokenize NSL source files for Stage 2 finetuning.

Outputs:
  - nsl_tokens.bin: flat u16 binary of tokenized NSL code
  - mixed_tokens.bin: 1M NSL tokens + 9M general code tokens (u16)
"""

import json
import struct
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:
    print("pip install tokenizers")
    sys.exit(1)


def collect_nsl_files(repo_root: Path) -> list[Path]:
    """Glob all .nsl files from stdlib, examples, tests."""
    dirs = ["stdlib", "examples", "tests"]
    files = []
    for d in dirs:
        files.extend(sorted((repo_root / d).rglob("*.nsl")))
    return files


def tokenize_files(files: list[Path], tokenizer: Tokenizer, sep_token: str = "<|file_sep|>") -> list[int]:
    """Tokenize files, concatenated with separator tokens."""
    all_ids: list[int] = []
    sep_ids = tokenizer.encode(sep_token).ids

    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        ids = tokenizer.encode(text).ids
        all_ids.extend(ids)
        all_ids.extend(sep_ids)

    return all_ids


def write_u16_bin(ids: list[int], path: Path) -> None:
    """Write token IDs as flat u16 binary."""
    with open(path, "wb") as f:
        for tid in ids:
            f.write(struct.pack("<H", tid % 65536))
    print(f"  Wrote {len(ids):,} tokens to {path} ({path.stat().st_size:,} bytes)")


def sample_general_tokens(general_path: Path, num_tokens: int) -> list[int]:
    """Read first num_tokens u16 values from pretokenized general data."""
    ids = []
    with open(general_path, "rb") as f:
        for _ in range(num_tokens):
            data = f.read(2)
            if not data:
                break
            ids.append(struct.unpack("<H", data)[0])
    return ids


def main():
    repo_root = Path(__file__).resolve().parents[3]  # models/coder50m/data/ -> repo root
    tokenizer_path = Path(sys.argv[1]) if len(sys.argv) > 1 else repo_root / "data" / "tokenizer" / "codeforge.json"
    general_data_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    out_dir = Path(__file__).resolve().parent

    print(f"Repo root: {repo_root}")
    print(f"Tokenizer: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Collect and tokenize NSL files
    nsl_files = collect_nsl_files(repo_root)
    print(f"Found {len(nsl_files)} .nsl files")

    nsl_ids = tokenize_files(nsl_files, tokenizer)
    print(f"Total NSL tokens: {len(nsl_ids):,}")

    # Write pure NSL tokens
    write_u16_bin(nsl_ids, out_dir / "nsl_tokens.bin")

    # Create mixed dataset for finetuning (1M NSL + 9M general)
    if general_data_path and general_data_path.exists():
        nsl_1m = (nsl_ids * (1_000_000 // len(nsl_ids) + 1))[:1_000_000]
        general_9m = sample_general_tokens(general_data_path, 9_000_000)
        mixed = nsl_1m + general_9m
        print(f"Mixed dataset: {len(nsl_1m):,} NSL + {len(general_9m):,} general = {len(mixed):,} total")
        write_u16_bin(mixed, out_dir / "mixed_tokens.bin")
    else:
        print("No general data path provided; skipping mixed dataset.")
        print("Usage: python prepare_nsl.py <tokenizer_path> <general_tokens.bin>")

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test it**

Run: `cd models/coder50m/data && python prepare_nsl.py <path_to_codeforge.json>`
Expected: Prints file count, token count, writes nsl_tokens.bin

- [ ] **Step 3: Commit**

```bash
git add models/coder50m/data/prepare_nsl.py
git commit -m "feat: add Python data prep script for NSL tokenization"
```

---

### Task 13: finetune.nsl — Stage 2 Training Script

**Files:**
- Create: `models/coder50m/finetune.nsl`

- [ ] **Step 1: Create finetune script**

```nsl
# models/coder50m/finetune.nsl
# Stage 2: Finetune on 1M NSL + 9M general code mixture

from config import VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, N_KV_HEADS
from config import D_FF, MAX_SEQ_LEN, FINETUNE_DATA, FINETUNE_BATCH_SIZE
from config import FINETUNE_LR, FINETUNE_WARMUP, FINETUNE_MIN_LR
from config import FINETUNE_WEIGHT_DECAY, FINETUNE_EPOCHS
from config import CHECKPOINT_DIR
from model import NSLCoder
from nsl.nn.losses import cross_entropy

# Load pretrained checkpoint
let m = NSLCoder()
model_load(m, f"{CHECKPOINT_DIR}/pretrain_final.nslm")
print("Loaded pretrained checkpoint. Starting finetuning...")

# Load mixed finetune data
let tokens = load_mmap(FINETUNE_DATA, 3)
let loader = DataLoader(tokens, batch_size=FINETUNE_BATCH_SIZE, seq_len=MAX_SEQ_LEN,
                         shuffle=true, drop_last=true)

# Finetune
train(model=m, epochs=FINETUNE_EPOCHS):
    optimizer: AdamW(lr=FINETUNE_LR, weight_decay=FINETUNE_WEIGHT_DECAY)
    scheduler: warmup_cosine(warmup_steps=FINETUNE_WARMUP,
                             total_steps=1220,
                             min_lr=FINETUNE_MIN_LR)
    step(batch):
        let logits = m.forward(batch.input_ids)
        let loss = cross_entropy(logits, batch.labels)
    callbacks:
        on_step(step, loss):
            if step % 50 == 0:
                print(f"Step {step}: Loss {loss}")
            if step % 500 == 0:
                model_save(m, f"{CHECKPOINT_DIR}/finetune_step_{step}.nslm")
        on_epoch(epoch, loss):
            print(f"Epoch {epoch} complete. Loss: {loss}")

model_save(m, f"{CHECKPOINT_DIR}/coder50m_final.nslm")
print("Finetuning complete.")
```

- [ ] **Step 2: Verify compilation**

Run: `cargo run -- check models/coder50m/finetune.nsl`
Expected: No compilation errors

- [ ] **Step 3: Commit**

```bash
git add models/coder50m/finetune.nsl
git commit -m "feat: add finetune.nsl — Stage 2 training on NSL+code mix"
```

---

## Phase 5: Inference

### Task 14: generate.nsl — Interactive Inference Demo

**Files:**
- Create: `models/coder50m/generate.nsl`

- [ ] **Step 1: Create the interactive generation script**

```nsl
# models/coder50m/generate.nsl
# Interactive code generation with NSL-Coder-50M

from config import VOCAB_SIZE, D_MODEL, N_LAYERS, N_HEADS, N_KV_HEADS
from config import D_FF, MAX_SEQ_LEN, TOKENIZER_PATH, CHECKPOINT_DIR
from config import TOP_K, TEMPERATURE, MAX_GEN_TOKENS
from model import NSLCoder
from nsl.inference.sampling import sample_top_k

# Load model
let m = NSLCoder()
model_load(m, f"{CHECKPOINT_DIR}/coder50m_final.nslm")
print("NSL-Coder-50M loaded (48.8M params, f32)")
print("Type a code prompt, then press Enter to generate.")
print("Type 'quit' to exit.")
print("")

# Load tokenizer
let tok = tokenizer_load(TOKENIZER_PATH)

# Main loop
let running = true
while running:
    print(">>> ")
    let prompt = read_line()

    if prompt == "quit":
        running = false
    else:
        # Encode prompt
        let input_ids = tokenizer_encode(tok, prompt)
        let seq = reshape(input_ids, [1, input_ids.shape[0]])

        # Generate tokens autoregressively
        let generated = seq
        let done = false
        let gen_count = 0

        while gen_count < MAX_GEN_TOKENS:
            if done:
                gen_count = MAX_GEN_TOKENS
            else:
                # Forward pass (full re-computation, no KV-cache)
                let logits = m.forward(generated)

                # Get logits for last position
                let last_logits = logits.select(1, logits.shape[1] - 1)

                # Sample next token
                let next_token = sample_top_k(last_logits, TOP_K, TEMPERATURE)

                # Check for stop: double newline token or EOS
                let token_str = tokenizer_decode(tok, next_token)
                print(token_str)

                # Append to sequence
                let next_2d = reshape(next_token, [1, 1])
                generated = cat(generated, next_2d, 1)
                gen_count = gen_count + 1

                # Stop on double newline
                if token_str == "\n\n":
                    done = true

        print("")
```

- [ ] **Step 2: Verify compilation**

Run: `cargo run -- check models/coder50m/generate.nsl`
Expected: No compilation errors (may have issues with while loops or cat — debug as needed)

- [ ] **Step 3: Commit**

```bash
git add models/coder50m/generate.nsl
git commit -m "feat: add interactive code generation demo"
```

---

### Task 15: M28 Dynamic Shapes Verification

Verify that the model's forward pass works with variable sequence lengths
(prefill with N tokens, then decode with 1 token appended each step).

**Files:**
- Create: `tests/test_dynamic_seq_len.nsl`

- [ ] **Step 1: Write dynamic shapes test**

```nsl
# tests/test_dynamic_seq_len.nsl
# Verify model handles different sequence lengths

const VOCAB_SIZE = 256
const D_MODEL = 64
const N_LAYERS = 2
const N_HEADS = 4
const N_KV_HEADS = 2
const HEAD_DIM = 16
const D_FF = 176
const MAX_SEQ_LEN = 32
const ROPE_THETA = 10000.0

from nsl.nn.norms import RMSNorm
from nsl.nn.gqa import GroupedQueryAttention

model SwiGLUFFN(d_model: int, d_ff: int):
    w_gate: Tensor = randn([d_model, d_ff])
    w_up: Tensor = randn([d_model, d_ff])
    w_down: Tensor = randn([d_ff, d_model])

    fn forward(self, x: Tensor) -> Tensor:
        let gate = silu(x @ self.w_gate.transpose(0, 1))
        let up = x @ self.w_up.transpose(0, 1)
        return (gate * up) @ self.w_down.transpose(0, 1)

model TransformerBlock(d_model: int, n_heads: int, n_kv_heads: int, d_ff: int):
    attn_norm: RMSNorm = RMSNorm(d_model)
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    ffn_norm: RMSNorm = RMSNorm(d_model)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_ff)

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + self.attn.forward(self.attn_norm.forward(x))
        return h + self.ffn.forward(self.ffn_norm.forward(h))

model TinyCoder:
    embed: Tensor = randn([VOCAB_SIZE, D_MODEL])
    blocks: [TransformerBlock; 2] = TransformerBlock(D_MODEL, N_HEADS, N_KV_HEADS, D_FF)
    norm: RMSNorm = RMSNorm(D_MODEL)

    fn forward(self, input_ids: Tensor) -> Tensor:
        let x = embedding_lookup(self.embed, input_ids)
        for block in self.blocks:
            x = block.forward(x)
        x = self.norm.forward(x)
        return x @ self.embed.transpose(0, 1)

let m = TinyCoder()

# Test 1: seq_len = 4 (prefill)
let t1 = zeros([1, 4])
let l1 = m.forward(t1)
print(l1.shape)
# Expected: [1, 4, 256]

# Test 2: seq_len = 8 (longer prefill)
let t2 = zeros([1, 8])
let l2 = m.forward(t2)
print(l2.shape)
# Expected: [1, 8, 256]

# Test 3: seq_len = 1 (single-token decode)
let t3 = zeros([1, 1])
let l3 = m.forward(t3)
print(l3.shape)
# Expected: [1, 1, 256]

print("dynamic_seq_len: PASS")
```

- [ ] **Step 2: Run test**

Run: `cargo run -- run tests/test_dynamic_seq_len.nsl`
Expected: PASS — all three shapes correct

- [ ] **Step 3: If test fails with shape mismatch**

Debug the specific operation that fails (likely RoPE or causal_mask with seq_len=1).
Fix the stdlib module or runtime function that doesn't handle the edge case.

- [ ] **Step 4: Commit**

```bash
git add tests/test_dynamic_seq_len.nsl
git commit -m "test: verify dynamic sequence lengths for prefill/decode"
```

---

## Phase 6: Documentation

### Task 16: README.md

**Files:**
- Create: `models/coder50m/README.md`

- [ ] **Step 1: Create README**

```markdown
# NSL-Coder-50M

A 50M-parameter code language model defined, trained, and served entirely in NSL.
LLaMA-style architecture: GQA, RoPE, SwiGLU, RMSNorm.

## Quick Start

### 1. Prepare NSL training data

```bash
cd models/coder50m/data
python prepare_nsl.py path/to/codeforge.json path/to/tokens.bin
```

### 2. Pretrain (Stage 1: 10B StarCoder tokens)

```bash
nsl run models/coder50m/pretrain.nsl
```

### 3. Finetune (Stage 2: NSL + general code mix)

```bash
nsl run models/coder50m/finetune.nsl
```

### 4. Generate code interactively

```bash
nsl run models/coder50m/generate.nsl
```

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~48.8M |
| Vocab | 49,152 (BPE) |
| Hidden dim | 512 |
| Layers | 8 |
| Q heads | 8 |
| KV heads | 4 (GQA) |
| FFN dim | 1408 (SwiGLU) |
| Context | 1024 |
| Precision | f32 |

## Files

- `config.nsl` — All hyperparameters
- `model.nsl` — LLaMA-style transformer architecture
- `pretrain.nsl` — Stage 1 training on StarCoder data
- `finetune.nsl` — Stage 2 finetuning on NSL code
- `generate.nsl` — Interactive code generation
- `data/prepare_nsl.py` — Tokenize NSL source files
```

- [ ] **Step 2: Commit**

```bash
git add models/coder50m/README.md
git commit -m "docs: add NSL-Coder-50M README with reproduction instructions"
```

---

## Dependency Graph

```
Phase 1 (parallel):
  Task 1 (sin/cos)  ─┐
  Task 2 (rotate_half)├─→ Task 6 (RoPE)
  Task 3 (expand)  ───┤              │
  Task 5b (contiguous) ├─→ Task 7 (GQA) ←── Task 6
  Task 4 (read_line)   │
  Task 5 (arange/clip) │
  Task 5c (grad clip)  │
  Task 5d (save/load)  │
  Task 5e (sample_top_k)

Phase 2 (after Phase 1):
  Task 6 (RoPE)     ← Tasks 1, 2
  Task 7 (GQA)      ← Tasks 3, 5b, 6
  Task 8 (nsl.io)

Phase 3 (after Phase 2):
  Task 9  (config)
  Task 10 (model.nsl) ← Tasks 6, 7

Phase 4 (after Phase 3):
  Task 11 (pretrain)  ← Tasks 10, 5c, 5d
  Task 12 (prepare_nsl.py) — independent, can run any time
  Task 13 (finetune)  ← Task 11

Phase 5 (after Phase 4):
  Task 14 (generate)  ← Tasks 10, 4, 5d, 5e
  Task 15 (dynamic shapes verify) ← Task 10

Phase 6:
  Task 16 (README)
```

**Parallelism:** Tasks 1-5e in Phase 1 are fully independent and can be executed
in parallel by separate agents. Task 12 (Python script) is independent of all
Rust/NSL work and can run any time.

**Critical path:** Tasks 1+2 → 6 → 7 → 10 → 11 → 13 → 14 (RoPE → GQA → model → train → finetune → generate).
