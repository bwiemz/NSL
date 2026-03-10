# M15: Tokenization + Standard Library Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Full tokenization system, neural network standard library, and test framework — culminating in "tokenize text → train a language model" end-to-end.

**Architecture:** Bottom-up, runtime-first. Phase 1 adds Rust runtime primitives. Phase 2 builds the test framework. Phase 3 implements nn layers in NSL. Phase 4 builds the tokenizer system. Phase 5 wires everything into the integration demo.

**Tech Stack:** Rust (nsl-runtime), Cranelift (codegen), HuggingFace `tokenizers` crate, NSL stdlib

**Design doc:** `docs/plans/2026-03-10-m15-tokenization-stdlib-design.md`

---

## Phase 1: Runtime Primitives

### Task 1: Tensor randn + Training Mode

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-semantic/src/builtins.rs`

**Step 1: Implement `nsl_tensor_randn` in tensor.rs**

Add after `nsl_tensor_rand`:

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_randn(shape_list: i64) -> i64 {
    let list = NslList::from_ptr(shape_list);
    let ndim = list.len as usize;
    let mut total: i64 = 1;
    let shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    let strides = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim {
        let dim = unsafe { *list.data.add(i) };
        unsafe { *shape.add(i) = dim; }
        total *= dim;
    }
    // Compute strides (row-major)
    let mut stride = 1i64;
    for i in (0..ndim).rev() {
        unsafe { *strides.add(i) = stride; }
        stride *= unsafe { *shape.add(i) };
    }
    let data = checked_alloc((total as usize) * std::mem::size_of::<f64>()) as *mut f64;
    // Box-Muller transform for normal distribution
    use std::f64::consts::PI;
    let mut i = 0usize;
    while i < total as usize {
        let u1: f64 = (rand_u64() as f64) / (u64::MAX as f64);
        let u2: f64 = (rand_u64() as f64) / (u64::MAX as f64);
        let u1 = if u1 < 1e-10 { 1e-10 } else { u1 }; // avoid log(0)
        let mag = (-2.0 * u1.ln()).sqrt();
        unsafe { *data.add(i) = mag * (2.0 * PI * u2).cos(); }
        i += 1;
        if i < total as usize {
            unsafe { *data.add(i) = mag * (2.0 * PI * u2).sin(); }
            i += 1;
        }
    }
    let tensor = Box::new(NslTensor { data, shape, strides, ndim: ndim as i64, len: total });
    let ptr = Box::into_raw(tensor) as i64;
    crate::autodiff::maybe_record_leaf(ptr);
    ptr
}
```

Note: `rand_u64()` is a helper using the existing PRNG from `nsl_tensor_rand`. Extract it if not already a helper.

**Step 2: Implement training mode global state**

Add a new file or section in `tensor.rs` (or a dedicated `training.rs`):

```rust
use std::cell::Cell;

thread_local! {
    static TRAINING_MODE: Cell<bool> = Cell::new(false);
}

#[no_mangle]
pub extern "C" fn nsl_set_training_mode(mode: i8) {
    TRAINING_MODE.with(|t| t.set(mode != 0));
}

#[no_mangle]
pub extern "C" fn nsl_is_training() -> i8 {
    TRAINING_MODE.with(|t| if t.get() { 1 } else { 0 })
}
```

**Step 3: Register in builtins**

In `crates/nsl-codegen/src/builtins.rs`, add to RUNTIME_FUNCTIONS:
```rust
("nsl_tensor_randn", &[types::I64], Some(types::I64)),
("nsl_set_training_mode", &[types::I8], None),
("nsl_is_training", &[], Some(types::I8)),
```

In `crates/nsl-semantic/src/builtins.rs`, add:
```rust
def("randn", Type::Function { params: vec![Type::List(Box::new(Type::Int))], ret: Box::new(Type::Tensor { shape: None, dtype: None }) });
def("set_training", Type::Function { params: vec![Type::Bool], ret: Box::new(Type::Void) });
def("is_training", Type::Function { params: vec![], ret: Box::new(Type::Bool) });
```

**Step 4: Wire `randn` dispatch in codegen expr.rs**

Follow the same pattern as `zeros`/`ones`/`rand` — add a match arm for `"randn"` in the function call dispatch.

**Step 5: Test**

Create `examples/m15_randn_test.nsl`:
```python
let t = randn([3, 4])
print(t)
```

Run: `cargo run -p nsl-cli -- run examples/m15_randn_test.nsl`
Expected: prints a 3×4 tensor with approximately normal values (mean ~0, values between -3 and 3 typically)

**Step 6: Commit**
```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-codegen/src/builtins.rs crates/nsl-semantic/src/builtins.rs crates/nsl-codegen/src/expr.rs examples/m15_randn_test.nsl
git commit -m "feat(m15): add randn tensor creation and training mode global state"
```

---

### Task 2: Activation Functions (Runtime + Autodiff)

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-runtime/src/autodiff.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-semantic/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

**Step 1: Implement activation forward passes in tensor.rs**

Add after existing element-wise ops (after `nsl_tensor_clamp`):

```rust
#[no_mangle]
pub extern "C" fn nsl_tensor_relu(tensor_ptr: i64) -> i64 {
    let t = NslTensor::from_ptr(tensor_ptr);
    let out = alloc_same_shape(t);
    for i in 0..t.len as usize {
        let v = unsafe { *t.data.add(i) };
        unsafe { *out.data.add(i) = if v > 0.0 { v } else { 0.0 }; }
    }
    let out_ptr = Box::into_raw(Box::new(out)) as i64;
    record_unary_op(tensor_ptr, out_ptr, TapeOp::ReLU);
    out_ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_gelu(tensor_ptr: i64) -> i64 {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let t = NslTensor::from_ptr(tensor_ptr);
    let out = alloc_same_shape(t);
    let sqrt_2_pi: f64 = (2.0 / std::f64::consts::PI).sqrt();
    for i in 0..t.len as usize {
        let x = unsafe { *t.data.add(i) };
        let inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        unsafe { *out.data.add(i) = 0.5 * x * (1.0 + inner.tanh()); }
    }
    let out_ptr = Box::into_raw(Box::new(out)) as i64;
    record_unary_with_saved(tensor_ptr, out_ptr, "gelu");
    out_ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_silu(tensor_ptr: i64) -> i64 {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    let t = NslTensor::from_ptr(tensor_ptr);
    let out = alloc_same_shape(t);
    for i in 0..t.len as usize {
        let x = unsafe { *t.data.add(i) };
        let sig = 1.0 / (1.0 + (-x).exp());
        unsafe { *out.data.add(i) = x * sig; }
    }
    let out_ptr = Box::into_raw(Box::new(out)) as i64;
    record_unary_with_saved(tensor_ptr, out_ptr, "silu");
    out_ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sigmoid(tensor_ptr: i64) -> i64 {
    let t = NslTensor::from_ptr(tensor_ptr);
    let out = alloc_same_shape(t);
    for i in 0..t.len as usize {
        let x = unsafe { *t.data.add(i) };
        unsafe { *out.data.add(i) = 1.0 / (1.0 + (-x).exp()); }
    }
    let out_ptr = Box::into_raw(Box::new(out)) as i64;
    record_unary_with_saved_out(tensor_ptr, out_ptr, "sigmoid");
    out_ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_tanh_act(tensor_ptr: i64) -> i64 {
    let t = NslTensor::from_ptr(tensor_ptr);
    let out = alloc_same_shape(t);
    for i in 0..t.len as usize {
        let x = unsafe { *t.data.add(i) };
        unsafe { *out.data.add(i) = x.tanh(); }
    }
    let out_ptr = Box::into_raw(Box::new(out)) as i64;
    record_unary_with_saved_out(tensor_ptr, out_ptr, "tanh");
    out_ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_softmax(tensor_ptr: i64, dim: i64) -> i64 {
    // Numerically stable: subtract max, then exp, then normalize
    let t = NslTensor::from_ptr(tensor_ptr);
    let max_t = nsl_tensor_reduce_max(tensor_ptr, dim, 1);  // keepdim=true
    let shifted = nsl_tensor_sub(tensor_ptr, max_t);
    let exp_t = nsl_tensor_exp(shifted);
    let sum_t = nsl_tensor_sum_dim(shifted, dim, 1);  // sum of exp
    // Actually: exp each element, sum, divide
    // Better: implement directly for efficiency
    // ... (full implementation handles the dim reduction properly)
    todo!("implement with proper dim handling")
}
```

Note: `softmax` needs careful implementation with proper dimensional handling. The implementer should write a direct loop version that handles the dim parameter correctly.

Helper functions needed: `alloc_same_shape(t: &NslTensor) -> NslTensor` (allocate output with same shape/strides).

**Step 2: Add TapeOp variants in autodiff.rs**

Add to the `TapeOp` enum (after `Clamp`):
```rust
ReLU { a: i64, out: i64, saved_a: i64 },
GELU { a: i64, out: i64, saved_a: i64 },
SiLU { a: i64, out: i64, saved_a: i64 },
Sigmoid { a: i64, out: i64, saved_out: i64 },
Tanh { a: i64, out: i64, saved_out: i64 },
Softmax { a: i64, out: i64, saved_out: i64, dim: i64 },
```

**Step 3: Implement backward passes in autodiff.rs**

Add match arms in the backward pass (after `Clamp`):
```rust
TapeOp::ReLU { a, out, saved_a } => {
    if let Some(&g) = grad_map.get(out) {
        // grad = g * (saved_a > 0)
        let sa = NslTensor::from_ptr(*saved_a);
        let g_t = NslTensor::from_ptr(g);
        let grad = alloc_same_shape(sa);
        for i in 0..sa.len as usize {
            let v = unsafe { *sa.data.add(i) };
            let gv = unsafe { *g_t.data.add(i) };
            unsafe { *grad.data.add(i) = if v > 0.0 { gv } else { 0.0 }; }
        }
        let grad_ptr = Box::into_raw(Box::new(grad)) as i64;
        accumulate_grad(&mut grad_map, *a, grad_ptr);
    }
}
// Sigmoid: d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) = out * (1 - out)
TapeOp::Sigmoid { a, out, saved_out } => {
    if let Some(&g) = grad_map.get(out) {
        let so = NslTensor::from_ptr(*saved_out);
        let g_t = NslTensor::from_ptr(g);
        let grad = alloc_same_shape(so);
        for i in 0..so.len as usize {
            let s = unsafe { *so.data.add(i) };
            let gv = unsafe { *g_t.data.add(i) };
            unsafe { *grad.data.add(i) = gv * s * (1.0 - s); }
        }
        let grad_ptr = Box::into_raw(Box::new(grad)) as i64;
        accumulate_grad(&mut grad_map, *a, grad_ptr);
    }
}
// GELU, SiLU, Tanh: similar patterns with their respective derivatives
```

**Step 4: Register in builtins and wire dispatch**

Add to `crates/nsl-codegen/src/builtins.rs` RUNTIME_FUNCTIONS:
```rust
("nsl_tensor_relu", &[types::I64], Some(types::I64)),
("nsl_tensor_gelu", &[types::I64], Some(types::I64)),
("nsl_tensor_silu", &[types::I64], Some(types::I64)),
("nsl_tensor_sigmoid", &[types::I64], Some(types::I64)),
("nsl_tensor_tanh_act", &[types::I64], Some(types::I64)),
("nsl_tensor_softmax", &[types::I64, types::I64], Some(types::I64)),
```

Add to `crates/nsl-semantic/src/builtins.rs` and wire dispatch in `crates/nsl-codegen/src/expr.rs` following the pattern for `exp`/`log`/`sqrt`.

**Step 5: Test**

Create `examples/m15_activations_test.nsl`:
```python
let x = full([4], -1.0)
let r = relu(x)
print(r)

let x2 = ones([2, 3])
let g = gelu(x2)
print(g)
```

Run: `cargo run -p nsl-cli -- run examples/m15_activations_test.nsl`
Expected: relu of [-1,-1,-1,-1] → [0,0,0,0]; gelu of ones → approximately [0.841, ...]

**Step 6: Commit**
```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-runtime/src/autodiff.rs crates/nsl-codegen/src/builtins.rs crates/nsl-semantic/src/builtins.rs crates/nsl-codegen/src/expr.rs examples/m15_activations_test.nsl
git commit -m "feat(m15): add activation functions with autodiff (relu, gelu, silu, sigmoid, tanh, softmax)"
```

---

### Task 3: Tensor Slice + Cat

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-runtime/src/autodiff.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-semantic/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

**Step 1: Implement `nsl_tensor_slice` in tensor.rs**

```rust
/// Slice a tensor along a dimension: tensor[start:end] on dim
#[no_mangle]
pub extern "C" fn nsl_tensor_slice(tensor_ptr: i64, dim: i64, start: i64, end: i64) -> i64 {
    let t = NslTensor::from_ptr(tensor_ptr);
    let d = dim as usize;
    let dim_size = unsafe { *t.shape.add(d) };
    let s = if start < 0 { dim_size + start } else { start };
    let e = if end < 0 { dim_size + end } else { end };
    let slice_len = e - s;
    // Build output shape (same as input but dim d has slice_len)
    // Copy data with proper stride handling
    // Record TapeOp::Slice for backward
    todo!("full implementation with stride-aware copy")
}
```

**Step 2: Implement `nsl_tensor_cat` in tensor.rs**

```rust
/// Concatenate a list of tensors along a dimension
#[no_mangle]
pub extern "C" fn nsl_tensor_cat(tensor_list: i64, dim: i64) -> i64 {
    let list = NslList::from_ptr(tensor_list);
    // Validate all tensors have same shape except on cat dim
    // Allocate output with summed size on cat dim
    // Copy data from each input tensor
    // Record TapeOp::Cat for backward (stores split sizes)
    todo!("full implementation")
}
```

**Step 3: Add TapeOp variants and backward passes**

```rust
Slice { a: i64, out: i64, dim: i64, start: i64, input_shape: Vec<i64> },
Cat { inputs: Vec<i64>, out: i64, dim: i64, split_sizes: Vec<i64> },
```

Backward for Slice: pad with zeros at the sliced positions.
Backward for Cat: split the gradient along the cat dim.

**Step 4: Register builtins, wire dispatch, test**

Test with:
```python
let t = arange(0.0, 10.0, 1.0)
let s = slice(t, 0, 2, 7)
print(s)
```

**Step 5: Commit**
```bash
git commit -m "feat(m15): add tensor slice and cat operations with autodiff"
```

---

### Task 4: Embedding Lookup

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-runtime/src/autodiff.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-semantic/src/builtins.rs`

**Step 1: Implement `nsl_tensor_embedding_lookup` in tensor.rs**

```rust
/// Look up rows from an embedding weight matrix by integer indices
/// weight: [vocab_size, embed_dim], indices: [seq_len] (int tensor) → output: [seq_len, embed_dim]
#[no_mangle]
pub extern "C" fn nsl_tensor_embedding_lookup(weight_ptr: i64, indices_ptr: i64) -> i64 {
    let weight = NslTensor::from_ptr(weight_ptr);
    let indices = NslTensor::from_ptr(indices_ptr);
    let vocab_size = unsafe { *weight.shape.add(0) };
    let embed_dim = unsafe { *weight.shape.add(1) };
    let seq_len = indices.len;
    // Allocate output [seq_len, embed_dim]
    // For each index: copy row from weight into output
    // Record TapeOp::EmbeddingLookup { weight, indices, out }
    // Backward: scatter-add gradients into weight gradient
    todo!("full implementation")
}
```

**Step 2: Add TapeOp and backward**

```rust
EmbeddingLookup { weight: i64, indices: i64, out: i64 },
```

Backward: create zero gradient for weight, scatter-add output gradient rows at index positions.

**Step 3: Register, test, commit**
```bash
git commit -m "feat(m15): add embedding lookup with autodiff backward"
```

---

### Task 5: LayerNorm + RMSNorm (Fused)

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-runtime/src/autodiff.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-semantic/src/builtins.rs`

**Step 1: Implement `nsl_tensor_layernorm`**

```rust
/// LayerNorm: normalize last dim, apply weight and bias
/// input: [*, normalized_shape], weight: [normalized_shape], bias: [normalized_shape]
#[no_mangle]
pub extern "C" fn nsl_tensor_layernorm(input_ptr: i64, weight_ptr: i64, bias_ptr: i64, eps: f64) -> i64 {
    let input = NslTensor::from_ptr(input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);
    // Compute mean and variance over last dim
    // Normalize: (x - mean) / sqrt(var + eps)
    // Scale and shift: normalized * weight + bias
    // Save {input, mean, inv_std, weight, bias} to tape
    todo!("full implementation")
}
```

TapeOp:
```rust
LayerNorm { input: i64, weight: i64, bias: i64, out: i64, saved_mean: i64, saved_inv_std: i64 },
```

**Step 2: Implement `nsl_tensor_rmsnorm`**

```rust
/// RMSNorm: normalize by RMS of last dim, apply weight (no bias, no mean subtraction)
#[no_mangle]
pub extern "C" fn nsl_tensor_rmsnorm(input_ptr: i64, weight_ptr: i64, eps: f64) -> i64 {
    // rms = sqrt(mean(x²) + eps)
    // output = x / rms * weight
    // Save {input, rms, weight} to tape
    todo!("full implementation")
}
```

TapeOp:
```rust
RMSNorm { input: i64, weight: i64, out: i64, saved_rms: i64 },
```

**Step 3: Backward passes**

LayerNorm backward is complex — reference PyTorch's implementation. Key formulas:
- `d_normalized = d_out * weight`
- `d_var = sum(d_normalized * (x - mean) * -0.5 * (var + eps)^(-1.5))`
- `d_mean = sum(d_normalized * -inv_std)`
- `d_input = d_normalized * inv_std + d_var * 2 * (x - mean) / N + d_mean / N`

**Step 4: Register, test, commit**
```bash
git commit -m "feat(m15): add fused LayerNorm and RMSNorm with autodiff"
```

---

### Task 6: Dropout + Conv2d + MaxPool2d

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`
- Modify: `crates/nsl-runtime/src/autodiff.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-semantic/src/builtins.rs`

**Step 1: Implement `nsl_tensor_dropout`**

```rust
/// Dropout: randomly zero elements with probability p during training
/// Returns scaled tensor. Saves mask to tape for backward.
#[no_mangle]
pub extern "C" fn nsl_tensor_dropout(tensor_ptr: i64, p: f64, training: i8) -> i64 {
    let t = NslTensor::from_ptr(tensor_ptr);
    if training == 0 || p == 0.0 {
        return nsl_tensor_clone(tensor_ptr); // No-op during eval
    }
    let scale = 1.0 / (1.0 - p);
    // Generate random mask (1 where keep, 0 where drop)
    // Apply: output = input * mask * scale
    // Save TapeOp::Dropout { input, mask, p, out }
    todo!("full implementation")
}
```

**Step 2: Implement `nsl_tensor_conv2d`**

```rust
/// 2D convolution: input [N, C_in, H, W], weight [C_out, C_in, kH, kW], bias [C_out]
#[no_mangle]
pub extern "C" fn nsl_tensor_conv2d(
    input_ptr: i64, weight_ptr: i64, bias_ptr: i64,
    stride_h: i64, stride_w: i64, pad_h: i64, pad_w: i64
) -> i64 {
    // Standard im2col + matmul approach, or direct nested loop
    // Save input, weight for backward
    todo!("full implementation")
}
```

**Step 3: Implement `nsl_tensor_maxpool2d`**

```rust
/// MaxPool2d: input [N, C, H, W] → output [N, C, H_out, W_out]
/// Saves argmax indices for backward gradient routing
#[no_mangle]
pub extern "C" fn nsl_tensor_maxpool2d(
    input_ptr: i64, kernel_h: i64, kernel_w: i64, stride: i64, padding: i64
) -> i64 {
    // For each window: find max value and its index
    // Save TapeOp::MaxPool2d { input_shape, indices, ... }
    todo!("full implementation")
}
```

**Step 4: Add TapeOps and backward passes**

```rust
Dropout { a: i64, out: i64, saved_mask: i64, p: f64 },
Conv2d { input: i64, weight: i64, out: i64, saved_input: i64, saved_weight: i64, stride: (i64, i64), padding: (i64, i64) },
MaxPool2d { a: i64, out: i64, saved_indices: Vec<usize>, input_shape: Vec<i64>, kernel: (i64, i64), stride: i64, padding: i64 },
```

- Dropout backward: `grad_input = grad_output * mask / (1 - p)`
- Conv2d backward: full convolution (rotate kernel 180°, convolve with grad)
- MaxPool2d backward: scatter gradient to argmax positions

**Step 5: Register builtins, test, commit**
```bash
git commit -m "feat(m15): add dropout, conv2d, maxpool2d with autodiff tape ops"
```

---

### Task 7: String Deallocation

**Files:**
- Modify: `crates/nsl-runtime/src/string.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

**Step 1: Add `nsl_string_free` to string.rs**

```rust
#[no_mangle]
pub extern "C" fn nsl_string_free(ptr: i64) {
    if ptr == 0 { return; }
    unsafe {
        let _ = CString::from_raw(ptr as *mut c_char);
        // CString drops and frees the memory
    }
}
```

**Step 2: Register in builtins**

```rust
("nsl_string_free", &[types::I64], None),
```

Note: Full scope-based string deallocation codegen is deferred — for M15 we add the function and manually call it where needed (tokenizer decode). Automatic lifetime tracking is a future milestone.

**Step 3: Commit**
```bash
git commit -m "feat(m15): add nsl_string_free for dynamic string deallocation"
```

---

## Phase 2: Test Framework

### Task 8: Assert Runtime Functions

**Files:**
- Modify: `crates/nsl-runtime/src/lib.rs`
- Create: `crates/nsl-runtime/src/testing.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-semantic/src/builtins.rs`

**Step 1: Create `testing.rs`**

```rust
use crate::tensor::NslTensor;

#[no_mangle]
pub extern "C" fn nsl_assert(condition: i8, msg_ptr: i64, msg_len: i64) {
    if condition != 0 { return; }
    let msg = if msg_ptr != 0 && msg_len > 0 {
        unsafe {
            let slice = std::slice::from_raw_parts(msg_ptr as *const u8, msg_len as usize);
            std::str::from_utf8_unchecked(slice)
        }
    } else {
        "assertion failed"
    };
    eprintln!("ASSERTION FAILED: {}", msg);
    std::process::abort();
}

#[no_mangle]
pub extern "C" fn nsl_assert_eq_int(a: i64, b: i64, msg_ptr: i64, msg_len: i64) {
    if a == b { return; }
    let msg = extract_msg(msg_ptr, msg_len);
    eprintln!("ASSERTION FAILED: {} (expected {} == {})", msg, a, b);
    std::process::abort();
}

#[no_mangle]
pub extern "C" fn nsl_assert_eq_float(a: f64, b: f64, msg_ptr: i64, msg_len: i64) {
    if a == b { return; }
    let msg = extract_msg(msg_ptr, msg_len);
    eprintln!("ASSERTION FAILED: {} (expected {} == {})", msg, a, b);
    std::process::abort();
}

#[no_mangle]
pub extern "C" fn nsl_assert_close(
    a_ptr: i64, b_ptr: i64, rtol: f64, atol: f64, msg_ptr: i64, msg_len: i64
) {
    let a = NslTensor::from_ptr(a_ptr);
    let b = NslTensor::from_ptr(b_ptr);
    // Step 1: Check ndim
    if a.ndim != b.ndim {
        let msg = extract_msg(msg_ptr, msg_len);
        eprintln!("ASSERTION FAILED: {} (ndim mismatch: {} vs {})", msg, a.ndim, b.ndim);
        std::process::abort();
    }
    // Step 2: Check each dimension
    for i in 0..a.ndim as usize {
        let da = unsafe { *a.shape.add(i) };
        let db = unsafe { *b.shape.add(i) };
        if da != db {
            let msg = extract_msg(msg_ptr, msg_len);
            eprintln!("ASSERTION FAILED: {} (shape mismatch at dim {}: {} vs {})", msg, i, da, db);
            std::process::abort();
        }
    }
    // Step 3: Element-wise check
    for i in 0..a.len as usize {
        let va = unsafe { *a.data.add(i) };
        let vb = unsafe { *b.data.add(i) };
        if (va - vb).abs() > atol + rtol * vb.abs() {
            let msg = extract_msg(msg_ptr, msg_len);
            eprintln!("ASSERTION FAILED: {} (element {} differs: {} vs {}, tol={})",
                msg, i, va, vb, atol + rtol * vb.abs());
            std::process::abort();
        }
    }
}

fn extract_msg(ptr: i64, len: i64) -> &'static str {
    if ptr != 0 && len > 0 {
        unsafe {
            let slice = std::slice::from_raw_parts(ptr as *const u8, len as usize);
            std::str::from_utf8_unchecked(slice)
        }
    } else {
        "assertion failed"
    }
}
```

**Step 2: Add `pub mod testing;` to lib.rs**

**Step 3: Register in builtins (codegen and semantic)**

Codegen:
```rust
("nsl_assert", &[types::I8, types::I64, types::I64], None),
("nsl_assert_eq_int", &[types::I64, types::I64, types::I64, types::I64], None),
("nsl_assert_eq_float", &[types::F64, types::F64, types::I64, types::I64], None),
("nsl_assert_close", &[types::I64, types::I64, types::F64, types::F64, types::I64, types::I64], None),
```

Semantic: register `assert`, `assert_eq`, `assert_close` as builtin functions.

**Step 4: Wire assert dispatch in expr.rs**

When `assert(cond)` is called, emit the condition value + a string constant for the source location. When `assert_close(a, b, rtol, atol)` is called, emit all 4 args + a message string.

The compiler needs to emit string constants into the data segment. Check how `print` handles string literals — follow the same pattern for assert messages.

**Step 5: Test**

```python
let x = ones([3])
let y = ones([3])
assert_close(x, y, 0.001, 0.001)
print("assertions passed")
```

**Step 6: Commit**
```bash
git commit -m "feat(m15): add assert, assert_eq, assert_close runtime and codegen"
```

---

### Task 9: @test Decorator Recognition

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`

**Step 1: Semantic checker validates @test**

In `checker.rs`, in the `StmtKind::Decorated` match arm, when decorator name is "test":
- Verify the inner statement is `FnDef`
- Verify function has no parameters
- Verify return type is Void

**Step 2: Codegen tracks @test functions**

In `compiler.rs`, add `test_fns: Vec<String>` field to `Compiler` struct.
In the `StmtKind::Decorated` handler, when name == "test", record the function name.

**Step 3: Commit**
```bash
git commit -m "feat(m15): recognize @test decorator in semantic checker and codegen"
```

---

### Task 10: `nsl test` CLI Command (Orchestrator)

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`

**Step 1: Add `Test` subcommand to Clap**

```rust
/// Run tests in an NSL file
Test {
    /// The NSL test file
    file: PathBuf,
    /// Filter tests by name substring
    #[arg(long)]
    filter: Option<String>,
},
```

**Step 2: Implement test runner**

The test runner needs to:
1. Parse the file to discover `@test` functions (reuse existing parse pipeline)
2. Compile a binary that accepts `--run <test_name>` argument
   - The generated `main` checks argv for `--run`, calls only that function, exits 0
   - If no `--run`, prints list of test names and exits
3. For each test (optionally filtered), spawn child process with `--run <name>`
4. Capture stdout/stderr, exit code
5. Print summary: `N passed, M failed`

Key: the compiled binary needs a special `main` that dispatches by name. This requires modifying `compile_main` to optionally emit a test-dispatch harness instead of the normal main body.

**Step 3: Test the test runner**

Create `examples/m15_test_framework.nsl`:
```python
@test
fn test_ones():
    let x = ones([3])
    assert_close(x, ones([3]), 0.001, 0.001)

@test
fn test_zeros():
    let x = zeros([3])
    assert_close(x, zeros([3]), 0.001, 0.001)
```

Run: `cargo run -p nsl-cli -- test examples/m15_test_framework.nsl`
Expected:
```
test_ones ... PASS
test_zeros ... PASS
2 passed, 0 failed
```

**Step 4: Commit**
```bash
git commit -m "feat(m15): add nsl test CLI command with orchestrator pattern"
```

---

## Phase 3: Neural Network Standard Library

### Task 11: Activation Functions (NSL stdlib)

**Files:**
- Create: `stdlib/nsl/nn/activations.nsl`

**Step 1: Write activation functions**

```python
fn relu(x: Tensor) -> Tensor:
    return tensor_relu(x)

fn gelu(x: Tensor) -> Tensor:
    return tensor_gelu(x)

fn silu(x: Tensor) -> Tensor:
    return tensor_silu(x)

fn sigmoid(x: Tensor) -> Tensor:
    return tensor_sigmoid(x)

fn tanh_act(x: Tensor) -> Tensor:
    return tensor_tanh_act(x)

fn softmax(x: Tensor, dim: int) -> Tensor:
    return tensor_softmax(x, dim)
```

Note: These are thin wrappers. The actual names may need to match whatever builtin dispatch names the codegen uses (e.g., if the codegen dispatches `relu(x)` directly to `nsl_tensor_relu`, these wrappers may be unnecessary — check how imports + builtins interact).

**Step 2: Test with nsl test**

```python
from nsl.nn.activations import relu, gelu

@test
fn test_relu_positive():
    let x = full([3], 2.0)
    assert_close(relu(x), full([3], 2.0), 1e-6, 1e-6)

@test
fn test_relu_negative():
    let x = full([3], -1.0)
    assert_close(relu(x), zeros([3]), 1e-6, 1e-6)
```

**Step 3: Commit**
```bash
git commit -m "feat(m15): add nsl.nn.activations stdlib module"
```

---

### Task 12: Linear Layer

**Files:**
- Create: `stdlib/nsl/nn/layers.nsl`

**Step 1: Write Linear model**

```python
model Linear:
    w: Tensor
    b: Tensor

    fn forward(self, x: Tensor) -> Tensor:
        return x @ transpose(self.w, 0, 1) + self.b
```

Note: The model constructor needs to accept `(in_features: int, out_features: int)` and initialize:
- `self.w = randn([out_features, in_features]) * sqrt(2.0 / in_features)` (Kaiming)
- `self.b = zeros([out_features])`

This requires model constructors to support computed initialization. Check how M11's model constructors handle default values. If they only support literal expressions, the implementer may need to:
- Create Linear via a factory function: `fn make_linear(in_feat: int, out_feat: int) -> Linear`
- Or extend model constructors to support arbitrary init expressions

**Step 2: Test**

```python
@test
fn test_linear_shape():
    let lin = Linear(4, 8)  # or make_linear(4, 8)
    let x = ones([2, 4])
    let y = lin.forward(x)
    # y should be [2, 8]
    assert_eq(shape(y)[0], 2)
    assert_eq(shape(y)[1], 8)
```

**Step 3: Commit**
```bash
git commit -m "feat(m15): add Linear layer to nsl.nn.layers"
```

---

### Task 13: Embedding Layer

**Files:**
- Modify: `stdlib/nsl/nn/layers.nsl`

**Step 1: Add Embedding model**

```python
model Embedding:
    weight: Tensor

    fn forward(self, indices: Tensor) -> Tensor:
        return embedding_lookup(self.weight, indices)
```

Constructor: `self.weight = randn([vocab_size, embed_dim]) * 0.02`

**Step 2: Test**

```python
@test
fn test_embedding_lookup():
    let emb = Embedding(10, 4)  # vocab=10, dim=4
    let ids = tensor_from_list([0, 3, 7])  # need integer tensor creation
    let out = emb.forward(ids)
    # out should be [3, 4]
    assert_eq(shape(out)[0], 3)
    assert_eq(shape(out)[1], 4)
```

Note: May need `tensor_from_list` or similar to create integer index tensors from lists.

**Step 3: Commit**
```bash
git commit -m "feat(m15): add Embedding layer to nsl.nn.layers"
```

---

### Task 14: LayerNorm + RMSNorm (NSL models)

**Files:**
- Create: `stdlib/nsl/nn/norms.nsl`

**Step 1: Write LayerNorm and RMSNorm**

```python
model LayerNorm:
    weight: Tensor
    bias: Tensor
    eps: float

    fn forward(self, x: Tensor) -> Tensor:
        return layernorm(x, self.weight, self.bias, self.eps)

model RMSNorm:
    weight: Tensor
    eps: float

    fn forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, self.eps)
```

Constructor: `weight = ones([dim])`, `bias = zeros([dim])`, `eps = 1e-5`

**Step 2: Test**

```python
@test
fn test_layernorm():
    let ln = LayerNorm(4)
    let x = ones([2, 4])
    let y = ln.forward(x)
    # After normalization of uniform input, should still be ~zeros (mean=0 after norm) + bias
    # Actually: ones normalized → zero mean, zero var → 0/eps → 0 * weight + bias = bias
```

**Step 3: Commit**
```bash
git commit -m "feat(m15): add LayerNorm and RMSNorm to nsl.nn.norms"
```

---

### Task 15: Dropout (NSL model)

**Files:**
- Create: `stdlib/nsl/nn/dropout.nsl`

**Step 1: Write Dropout model**

```python
model Dropout:
    p: float

    fn forward(self, x: Tensor) -> Tensor:
        return dropout(x, self.p, is_training())
```

**Step 2: Test, commit**
```bash
git commit -m "feat(m15): add Dropout to nsl.nn.dropout"
```

---

### Task 16: Multi-Head Attention

**Files:**
- Create: `stdlib/nsl/nn/attention.nsl`

**Step 1: Write Attention model**

```python
from nsl.nn.layers import Linear
from nsl.nn.activations import softmax

model Attention:
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    num_heads: int
    head_dim: int

    fn forward(self, x: Tensor) -> Tensor:
        let q = self.q_proj.forward(x)
        let k = self.k_proj.forward(x)
        let v = self.v_proj.forward(x)
        # Scale dot-product attention
        let scale = sqrt(full([1], self.head_dim))
        let attn = softmax(q @ transpose(k, 0, 1) / scale, -1)
        let out = attn @ v
        return self.out_proj.forward(out)
```

Note: This is simplified single-batch attention without head splitting. Multi-head requires reshape operations that may need `nsl_tensor_reshape` + view management. The implementer should start with the simple version and add head splitting if reshape + transpose are working.

**Step 2: Test, commit**
```bash
git commit -m "feat(m15): add multi-head Attention to nsl.nn.attention"
```

---

### Task 17: MLP + Conv2d + MaxPool2d (NSL models)

**Files:**
- Modify: `stdlib/nsl/nn/layers.nsl`

**Step 1: Add MLP model**

```python
from nsl.nn.activations import gelu

model MLP:
    fc1: Linear
    fc2: Linear

    fn forward(self, x: Tensor) -> Tensor:
        let h = gelu(self.fc1.forward(x))
        return self.fc2.forward(h)
```

**Step 2: Add Conv2d and MaxPool2d models**

```python
model Conv2d:
    weight: Tensor
    bias: Tensor

    fn forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, self.bias, self.stride_h, self.stride_w, self.pad_h, self.pad_w)

model MaxPool2d:
    kernel_h: int
    kernel_w: int
    stride: int
    padding: int

    fn forward(self, x: Tensor) -> Tensor:
        return maxpool2d(x, self.kernel_h, self.kernel_w, self.stride, self.padding)
```

**Step 3: Test, commit**
```bash
git commit -m "feat(m15): add MLP, Conv2d, MaxPool2d to nsl.nn.layers"
```

---

## Phase 4: Tokenizer System

### Task 18: Add HF tokenizers Dependency

**Files:**
- Modify: `crates/nsl-runtime/Cargo.toml`

**Step 1: Add dependency**

```toml
[dependencies]
tokenizers = "0.21"
```

**Step 2: Verify build**

Run: `cargo build -p nsl-runtime`
Expected: compiles successfully with the tokenizers crate

Note: This is the first external dependency in nsl-runtime. The `tokenizers` crate is pure Rust.

**Step 3: Commit**
```bash
git commit -m "feat(m15): add huggingface tokenizers crate dependency to nsl-runtime"
```

---

### Task 19: Tokenizer Runtime Module

**Files:**
- Create: `crates/nsl-runtime/src/tokenizer.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

**Step 1: Create tokenizer.rs with core functions**

```rust
use tokenizers::Tokenizer;
use tokenizers::models::bpe::BPE;
use tokenizers::models::wordpiece::WordPiece;
use crate::tensor::NslTensor;
use crate::list::NslList;
use crate::memory::checked_alloc;

/// Opaque handle: Box<Tokenizer> stored as i64 pointer
fn store_tokenizer(tok: Tokenizer) -> i64 {
    Box::into_raw(Box::new(tok)) as i64
}

fn get_tokenizer(handle: i64) -> &'static mut Tokenizer {
    unsafe { &mut *(handle as *mut Tokenizer) }
}

/// Create a byte-level tokenizer (no training needed)
#[no_mangle]
pub extern "C" fn nsl_byte_tokenizer_new() -> i64 {
    let tok = Tokenizer::new(tokenizers::models::byte_level::ByteLevel::default());
    // Configure normalizer, pre-tokenizer if needed
    store_tokenizer(tok)
}

/// Train BPE tokenizer from corpus file
#[no_mangle]
pub extern "C" fn nsl_bpe_train(
    corpus_path_ptr: i64, corpus_path_len: i64,
    vocab_size: i64, min_freq: i64,
    special_tokens_ptr: i64  // NslList of string pointers
) -> i64 {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(corpus_path_ptr as *const u8, corpus_path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    // Use HF tokenizers API:
    // let mut trainer = BpeTrainer::builder().vocab_size(vocab_size as usize).min_frequency(min_freq as u32).build();
    // let mut tokenizer = Tokenizer::new(BPE::default());
    // tokenizer.train_from_files(&mut trainer, vec![path.to_string()]).unwrap();
    // store_tokenizer(tokenizer)
    todo!("full implementation using HF tokenizers API")
}

/// Load tokenizer from JSON file
#[no_mangle]
pub extern "C" fn nsl_tokenizer_load(path_ptr: i64, path_len: i64) -> i64 {
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    match Tokenizer::from_file(path) {
        Ok(tok) => store_tokenizer(tok),
        Err(e) => {
            eprintln!("nsl: tokenizer_load: failed to load '{}': {}", path, e);
            std::process::abort();
        }
    }
}

/// Save tokenizer to JSON file
#[no_mangle]
pub extern "C" fn nsl_tokenizer_save(handle: i64, path_ptr: i64, path_len: i64) {
    let tok = get_tokenizer(handle);
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    if let Err(e) = tok.save(path, false) {
        eprintln!("nsl: tokenizer_save: failed to save '{}': {}", path, e);
        std::process::abort();
    }
}

/// Encode text to token IDs → returns 1D i64 tensor
#[no_mangle]
pub extern "C" fn nsl_tokenizer_encode(handle: i64, text_ptr: i64, text_len: i64) -> i64 {
    let tok = get_tokenizer(handle);
    let text = unsafe {
        let slice = std::slice::from_raw_parts(text_ptr as *const u8, text_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    let encoding = match tok.encode(text, false) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("nsl: tokenizer_encode: {}", e);
            std::process::abort();
        }
    };
    let ids = encoding.get_ids();
    // Create 1D tensor with shape [seq_len]
    let len = ids.len() as i64;
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
    for (i, &id) in ids.iter().enumerate() {
        unsafe { *data.add(i) = id as f64; }
    }
    let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = len; }
    let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides = 1; }
    let tensor = Box::new(NslTensor { data, shape, strides, ndim: 1, len });
    Box::into_raw(tensor) as i64
}

/// Decode token IDs back to text → returns string pointer (caller must free)
#[no_mangle]
pub extern "C" fn nsl_tokenizer_decode(handle: i64, tensor_ptr: i64) -> i64 {
    let tok = get_tokenizer(handle);
    let t = NslTensor::from_ptr(tensor_ptr);
    let ids: Vec<u32> = (0..t.len as usize)
        .map(|i| unsafe { *t.data.add(i) } as u32)
        .collect();
    let text = match tok.decode(&ids, true) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("nsl: tokenizer_decode: {}", e);
            std::process::abort();
        }
    };
    let cstr = std::ffi::CString::new(text).unwrap();
    cstr.into_raw() as i64
}

/// Encode batch → returns NslList of [input_ids_tensor, attention_mask_tensor]
#[no_mangle]
pub extern "C" fn nsl_tokenizer_encode_batch(
    handle: i64, texts_list: i64, padding: i8, truncation: i8, max_len: i64
) -> i64 {
    // Encode each text, find max length, pad/truncate, build attention masks
    // Return NslList of [input_ids [batch, seq], attention_mask [batch, seq]]
    todo!("full implementation")
}

/// Get vocabulary size
#[no_mangle]
pub extern "C" fn nsl_tokenizer_vocab_size(handle: i64) -> i64 {
    let tok = get_tokenizer(handle);
    tok.get_vocab_size(true) as i64
}
```

**Step 2: Add `pub mod tokenizer;` to lib.rs**

**Step 3: Register all functions in builtins.rs**

**Step 4: Test with a simple encode/decode**

```python
# Simple test - byte tokenizer
let tok = byte_tokenizer_new()
let ids = tokenizer_encode(tok, "hello")
print(ids)
```

**Step 5: Commit**
```bash
git commit -m "feat(m15): add tokenizer runtime module wrapping HF tokenizers crate"
```

---

### Task 20: Tokenizer Keyword Codegen

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`
- Modify: `crates/nsl-codegen/src/stmt.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`

**Step 1: Semantic validation**

In `checker.rs`, replace the no-op `TokenizerDef` handler:
- Validate `algorithm` config value ∈ {bpe, wordpiece, sentencepiece, unigram, char, byte}
- Validate `vocab_size` is positive int
- Validate body sections
- Declare tokenizer name as a variable of type Tokenizer (or Unknown)

**Step 2: Codegen desugaring**

In `stmt.rs`, handle `StmtKind::TokenizerDef`:
1. Extract config: algorithm name, vocab_size, model_file
2. Based on algorithm:
   - "byte" → call `nsl_byte_tokenizer_new()`
   - "char" → call `nsl_char_tokenizer_new()`
   - "bpe" with model_file → call `nsl_tokenizer_load(path)`
   - "bpe" without model_file → the tokenizer needs training later (store config for now)
3. Store handle in local variable (same name as tokenizer)
4. Handle `special_tokens` section → call `nsl_tokenizer_add_special_tokens(handle, ...)`

**Step 3: Remove TokenizerDef from codegen no-op lists**

In `compiler.rs` line ~1031 and `stmt.rs` line ~190, remove `StmtKind::TokenizerDef` from the skip arms.

**Step 4: Wire method calls**

When `tok.encode(text)` is called, the codegen needs to recognize this as a method on a tokenizer handle and dispatch to `nsl_tokenizer_encode(handle, text_ptr, text_len)`.

This is the trickiest part — it depends on how method calls on non-model objects work in codegen. May need to check if the variable type is "Tokenizer" and dispatch accordingly, similar to how model methods are dispatched.

**Step 5: Test**

```python
tokenizer ByteTok(algorithm="byte", vocab_size=256):
    special_tokens:
        pad = "<pad>"

let tok = ByteTok()
let ids = tok.encode("hello world")
print(ids)
```

**Step 6: Commit**
```bash
git commit -m "feat(m15): implement tokenizer keyword codegen with config validation"
```

---

### Task 21: BPE Training Integration

**Files:**
- Modify: `crates/nsl-runtime/src/tokenizer.rs`

**Step 1: Complete BPE training implementation**

Using HF tokenizers API:
```rust
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::trainers::BpeTrainer;

pub extern "C" fn nsl_bpe_train(...) -> i64 {
    let mut trainer = BpeTrainer::builder()
        .vocab_size(vocab_size as usize)
        .min_frequency(min_freq as u32)
        .special_tokens(special_tokens_vec)
        .build();
    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(ByteLevel::default());
    tokenizer.train_from_files(&mut trainer, vec![path.to_string()])
        .unwrap_or_else(|e| { eprintln!("..."); std::process::abort(); });
    store_tokenizer(tokenizer)
}
```

**Step 2: Test BPE training**

Create a small corpus file `examples/corpus.txt`:
```
hello world this is a test
the quick brown fox jumps over the lazy dog
```

```python
tokenizer MyBPE(algorithm="bpe", vocab_size=100):
    special_tokens:
        pad = "<pad>"
        eos = "<eos>"

let tok = MyBPE()
tok.train("examples/corpus.txt")
let ids = tok.encode("hello world")
let decoded = tok.decode(ids)
print(decoded)
```

**Step 3: Commit**
```bash
git commit -m "feat(m15): complete BPE training integration with HF tokenizers"
```

---

### Task 22: NSL Tokenizer Stdlib Wrappers

**Files:**
- Create: `stdlib/nsl/tokenize/tokenizer.nsl`
- Create: `stdlib/nsl/tokenize/trainers.nsl`

**Step 1: Write thin wrappers**

These are only needed if the tokenizer keyword doesn't provide all functionality directly. They're for users who want programmatic access:

```python
# tokenizer.nsl
fn load_tokenizer(path: str) -> Tokenizer:
    return tokenizer_load(path)

fn save_tokenizer(tok: Tokenizer, path: str):
    tokenizer_save(tok, path)
```

**Step 2: Commit**
```bash
git commit -m "feat(m15): add nsl.tokenize stdlib wrappers"
```

---

## Phase 5: Integration

### Task 23: Training Mode in Train Block

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`

**Step 1: Emit set_training_mode in train block codegen**

In `compile_train_block`, at the start of the step body (before tape_start):
```rust
// Set training mode = true
let true_val = builder.ins().iconst(cl_types::I8, 1);
self.compile_call_by_name(builder, "nsl_set_training_mode", &[true_val])?;
```

After the step body (after callbacks):
```rust
// Set training mode = false
let false_val = builder.ins().iconst(cl_types::I8, 0);
self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;
```

**Step 2: Test, commit**
```bash
git commit -m "feat(m15): emit training mode toggle in train block codegen"
```

---

### Task 24: Integration Test — Tiny Language Model

**Files:**
- Create: `examples/m15_tiny_lm.nsl`
- Create: `tests/expected/m15_tiny_lm.txt` (or validate loss decreases)

**Step 1: Write the integration example**

```python
from nsl.nn.layers import Linear, Embedding
from nsl.nn.norms import LayerNorm
from nsl.nn.activations import gelu
from nsl.nn.losses import cross_entropy

tokenizer ByteTok(algorithm="byte", vocab_size=256):
    special_tokens:
        pad = "<pad>"

model TinyLM:
    embed: Embedding(256, 64)
    ln: LayerNorm(64)
    proj: Linear(64, 256)

    fn forward(self, x: Tensor) -> Tensor:
        let h = self.embed.forward(x)
        h = self.ln.forward(h)
        h = gelu(h)
        return self.proj.forward(h)

let m = TinyLM()
let tok = ByteTok()
let encoded = tok.encode("hello world this is a test")

train(model=m, epochs=10):
    optimizer: Adam(lr=0.001)
    step(batch):
        let input = encoded[0..-1]
        let targets = encoded[1..]
        let logits = m.forward(input)
        let loss = cross_entropy(logits, targets)
    callbacks:
        on_epoch(epoch, loss):
            print(loss)
```

**Step 2: Run and verify**

Run: `cargo run -p nsl-cli -- run examples/m15_tiny_lm.nsl`
Expected: 10 lines of decreasing loss values

**Step 3: Commit**
```bash
git commit -m "feat(m15): add tiny language model integration test"
```

---

### Task 25: Full Test Suite

**Files:**
- Create: `tests/m15_nn_test.nsl`
- Create: `tests/m15_tokenizer_test.nsl`

**Step 1: Write nn layer tests**

```python
from nsl.nn.layers import Linear, Embedding
from nsl.nn.norms import LayerNorm, RMSNorm
from nsl.nn.activations import relu, gelu, silu, sigmoid, softmax
from nsl.nn.dropout import Dropout

@test
fn test_relu_positive():
    assert_close(relu(full([3], 2.0)), full([3], 2.0), 1e-6, 1e-6)

@test
fn test_relu_negative():
    assert_close(relu(full([3], -1.0)), zeros([3]), 1e-6, 1e-6)

@test
fn test_sigmoid_zero():
    assert_close(sigmoid(zeros([3])), full([3], 0.5), 1e-6, 1e-6)

@test
fn test_linear_shape():
    let lin = Linear(4, 8)
    let out = lin.forward(ones([2, 4]))
    assert_eq(shape(out)[0], 2)
    assert_eq(shape(out)[1], 8)

@test
fn test_embedding_shape():
    let emb = Embedding(10, 4)
    let out = emb.forward(tensor_from_ints([0, 3, 7]))
    assert_eq(shape(out)[0], 3)
    assert_eq(shape(out)[1], 4)

@test
fn test_layernorm():
    let ln = LayerNorm(4)
    let out = ln.forward(ones([2, 4]))
    # Uniform input → normalized to 0 + bias(0) = 0
    assert_close(out, zeros([2, 4]), 1e-5, 1e-5)
```

**Step 2: Write tokenizer tests**

```python
@test
fn test_byte_encode_decode():
    let tok = byte_tokenizer_new()
    let ids = tokenizer_encode(tok, "hello")
    let decoded = tokenizer_decode(tok, ids)
    assert_eq(decoded, "hello")

@test
fn test_vocab_size():
    let tok = byte_tokenizer_new()
    assert_eq(tokenizer_vocab_size(tok), 256)
```

**Step 3: Run test suite**

Run: `cargo run -p nsl-cli -- test tests/m15_nn_test.nsl`
Run: `cargo run -p nsl-cli -- test tests/m15_tokenizer_test.nsl`
Expected: all tests pass

**Step 4: Commit**
```bash
git commit -m "feat(m15): add comprehensive nn and tokenizer test suites"
```

---

### Task 26: Update README + Docs

**Files:**
- Modify: `README.md`

**Step 1: Update milestone table**

Change M15 status from "Planned" to "Complete".

**Step 2: Add nn layers and tokenizer to features section**

**Step 3: Commit and push**
```bash
git commit -m "docs: update README for M15 completion"
git push
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| Phase 1 | 1-7 | Runtime primitives: randn, activations, slice/cat, embedding, norms, dropout/conv/pool, string free |
| Phase 2 | 8-10 | Test framework: assert functions, @test decorator, nsl test CLI |
| Phase 3 | 11-17 | NN stdlib: activations, Linear, Embedding, LayerNorm, RMSNorm, Dropout, Attention, MLP, Conv2d |
| Phase 4 | 18-22 | Tokenizer: HF dependency, runtime module, keyword codegen, BPE training, stdlib wrappers |
| Phase 5 | 23-26 | Integration: training mode, tiny LM demo, test suite, docs |

**Total: 26 tasks across 5 phases.**

Each task is independently testable and committable. The bottom-up order ensures each piece validates the layer below.
