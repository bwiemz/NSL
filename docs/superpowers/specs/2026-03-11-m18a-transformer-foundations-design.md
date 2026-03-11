# M18a Design Spec: Transformer Foundations

**Date:** 2026-03-11
**Status:** Approved
**Milestone:** M18a (split from M18 — interop deferred to M18b)
**Prerequisite:** M17 (GPU/CUDA) complete

## Overview

M18a adds the tensor operations, model composition features, and stdlib layers needed to define
and run a multi-layer GPT-2-style transformer in pure NSL — with no interop, no external weights,
no Python FFI. The deliverable is: define a transformer, initialize with random weights, run forward
pass, train on a toy sequence, verify correctness.

## Scope

### In Scope
- 5 new Rust runtime tensor ops: `unsqueeze`, `expand`, `stack`, `causal_mask`, `select`
- Autodiff backward passes for `unsqueeze`, `expand`, `stack`
- Nested model fields (model-typed fields in model definitions)
- Fixed-size model arrays (`layers: [TransformerBlock; 12]`)
- `for layer in self.layers` iteration with compile-time-known bounds
- Updated stdlib: multi-head attention, transformer block, position embeddings
- End-to-end test: multi-layer transformer forward + train on toy data

### Out of Scope (Deferred)
- `@tie_weights` — requires alias analysis to prevent double-free and autodiff overwrite
- `where(cond, x, y)` — requires bool tensor dtype across the full stack
- `masked_fill` — eliminated; additive masking via `causal_mask` + existing `+` operator
- All interop: py.call, DLPack, safetensors, ONNX, HuggingFace
- `nsl export` command
- KV-cache for autoregressive generation
- `topk`, `multinomial` (deferred to M19)

---

## Part 1: New Tensor Operations

### 1.1 Runtime Signatures

All ops are `#[no_mangle] pub extern "C"` in `nsl-runtime/src/tensor.rs`, following existing patterns.

```rust
// Insert a dimension of size 1 at position `dim`
// unsqueeze([3, 4], dim=1) -> [3, 1, 4]
fn nsl_tensor_unsqueeze(tensor: *mut NslTensor, dim: i64) -> *mut NslTensor

// Broadcast tensor to target shape (no-copy where dimensions already match)
// expand([1, 4], shape=[3, 4]) -> [3, 4] with data repeated
fn nsl_tensor_expand(tensor: *mut NslTensor, shape: *const i64, ndim: i64) -> *mut NslTensor

// Concatenate N tensors along a new dimension
// stack([a, b, c], dim=0) where each is [4, 5] -> [3, 4, 5]
fn nsl_tensor_stack(tensors: *const *mut NslTensor, count: i64, dim: i64) -> *mut NslTensor

// Select a single index along a dimension, removing that dimension
// select([3, 4, 5], dim=0, index=1) -> [4, 5]
fn nsl_tensor_select(tensor: *mut NslTensor, dim: i64, index: i64) -> *mut NslTensor

// Generate additive causal mask: lower triangle = 0.0, upper triangle = -1e9
// Always produces CPU f64. User calls .to(cuda) if needed.
fn nsl_tensor_causal_mask(seq_len: i64) -> *mut NslTensor
```

### 1.2 Implementation Details

**unsqueeze:** Allocates a new NslTensor with a new shape array (ndim+1 dimensions), inserts size 1
at position `dim`. The data is copied to a new allocation (not shared) to keep the invariant that
every NslTensor owns its data independently — this simplifies autodiff and avoids aliasing issues.
Negative `dim` values are supported (Python-style: -1 = last dimension).

**expand:** For each dimension where the source size is 1 and target size is > 1, the data is
physically replicated. For dimensions that already match, no copy occurs. The result is always a
new tensor with its own data (no stride tricks — our tensors are contiguous-only).

Implementation note: Because NSL tensors are always contiguous (no stride-based views), `expand`
must physically copy data. This is simpler than PyTorch's lazy expand but uses more memory. For
M18a this is acceptable; stride-based views are a future optimization.

**stack:** Allocates output tensor with shape `[count, ...input_shape]` (for dim=0) or with the
new dimension inserted at `dim`. Copies data from each input tensor into the appropriate region.
All input tensors must have identical shapes.

**select:** Extracts a single slice along dimension `dim` at position `index`, removing that
dimension from the result. E.g., `select([3, 4, 5], dim=0, index=1)` returns a `[4, 5]` tensor.
Copies the relevant data region. Needed for `stack` backward (splitting gradients back to inputs).
Negative dim/index supported.

**causal_mask:** Generates a `[seq_len, seq_len]` tensor directly in a tight Rust loop:
```rust
for i in 0..seq_len {
    for j in 0..seq_len {
        data[i * seq_len + j] = if j <= i { 0.0 } else { -1e9 };
    }
}
```
Always produces CPU f64. User calls `.to(cuda)` if GPU is needed.
No autodiff — this is a constant tensor (not recorded on tape).

### 1.3 Device/Dtype Awareness

All new ops must follow the M17 convention:
- Check `tensor.dtype` (0=f64, 1=f32) and branch on data pointer type
- Propagate `device` and `dtype` from input to output
- For `causal_mask`: always produce CPU f64 (user calls `.to(cuda)` if needed)

### 1.4 Autodiff Backward Passes

New `TapeOp` variants in `nsl-runtime/src/autodiff.rs`:

```rust
TapeOp::Unsqueeze { result: TensorId, input: TensorId, dim: i64 }
// Backward: reshape grad_output to remove the inserted dim-1 axis.
// Since the inserted dimension has size 1, this is a pure reshape
// (no summation needed). Use nsl_tensor_reshape with the original
// input shape.

TapeOp::Expand { result: TensorId, input: TensorId, original_shape: Vec<i64> }
// Backward: reduce_sum along each broadcast dimension to collapse
// grad_output back to original_shape. Iterate dimensions in reverse
// order, calling sum_dim(keepdim=true) for each dim where
// original_shape[d] == 1 and output_shape[d] > 1.
// Multiple broadcast dims require sequential sum_dim calls.

TapeOp::Stack { result: TensorId, inputs: Vec<TensorId>, dim: i64 }
// Backward: select along stacked dimension to split gradient.
// For each i: grad_inputs[i] = nsl_tensor_select(grad_output, dim, i)
// This extracts and distributes gradient slices to each input.
```

`causal_mask` is NOT recorded on the tape — it's a constant (no gradient flows through it).

### 1.5 Codegen Integration

Register new functions in `builtins.rs`:
```rust
("nsl_tensor_unsqueeze",   &[types::I64, types::I64], Some(types::I64)),
("nsl_tensor_expand",      &[types::I64, types::I64, types::I64], Some(types::I64)),
("nsl_tensor_stack",       &[types::I64, types::I64, types::I64], Some(types::I64)),
("nsl_tensor_select",      &[types::I64, types::I64, types::I64], Some(types::I64)),
("nsl_tensor_causal_mask", &[types::I64], Some(types::I64)),
```

Compiler recognizes these as builtin function calls (same pattern as `zeros`, `reshape`, etc.).
`unsqueeze` is exposed as a method: `tensor.unsqueeze(dim)` via `compile_member_access`.
`expand` is exposed as a method: `tensor.expand([3, 4, 5])`. The list literal is converted
to a pointer+length pair during codegen, same pattern as `reshape([3, 4])`.
`stack` is a free function: `stack([a, b, c], dim=0)`.
`select` is exposed as a method: `tensor.select(dim, index)`.
`causal_mask` is a free function: `causal_mask(seq_len)`.

---

## Part 2: Nested Model Fields

### 2.1 Problem

Currently, model fields can only be primitive types or tensors. The type resolver maps all
non-primitive type names to `cl_types::I64` (opaque pointer) but the constructor doesn't know
how to initialize a model-typed field.

For a transformer, we need:
```nsl
model TransformerBlock(dim: int):
    attn: Attention = Attention(dim, 4)        # <-- nested model
    norm: LayerNorm = LayerNorm(dim)           # <-- nested model (stdlib)
    mlp: MLP = MLP(dim, dim * 4, dim)         # <-- nested model
```

### 2.2 Design

**All model fields are opaque pointers (I64).** This is already the case — model constructors
return `*mut u8` (an allocated pointer). A nested model field simply stores a pointer to
the sub-model's allocated memory. No inline embedding needed.

This means:
- **No struct layout changes.** A model field typed as another model is just an I64 (pointer).
  The offset calculation stays the same.
- **Constructor initializes via sub-constructor call.** When the field init expression is
  `Attention(dim, 4)`, the compiler calls `__nsl_model_Attention(dim, 4)` and stores the
  returned pointer at the field's offset.
- **Field access returns a pointer.** When compiling `self.attn`, the load produces an I64
  (pointer to sub-model), which is then used as `self` when calling `self.attn.forward(x)`.

### 2.3 Semantic Checker Changes

In `check_model_def` (checker.rs):
- When resolving a field's type annotation, if the type name matches a known model, resolve it
  to `Type::Model { name, fields, methods }` instead of erroring
- The init expression `Attention(dim, 4)` already type-checks as a model constructor call
- Method access on the field (`self.attn.forward(x)`) already works via
  `compile_member_access` → `compile_model_method_call` chain, as long as the intermediate
  type is correctly resolved as `Type::Model`

### 2.4 Codegen Changes

Minimal changes needed:
1. `resolve_type_name_to_cl()` — already maps unknown names to `I64` (pointer). No change.
2. `compile_model_constructor()` — field init expressions are already compiled generically.
   If `init` is `Attention(dim, 4)`, the existing expression compiler handles it as a function
   call that returns I64. Store at offset. **No change needed.**
3. `compile_member_access()` — when accessing `self.attn`, the load returns I64 (pointer).
   If we then access `.forward()`, the method call compilation needs the sub-model's type
   to find the mangled method name. **Change needed:** propagate Model type info through
   chained member access.

### 2.5 Chained Method Call Resolution

The key change is in `compile_member_access` (expr.rs). Currently:
```
self.attn           → load I64 from offset → returns Value
self.attn.forward(x) → ???
```

The chain `self.attn.forward(x)` is parsed as `Call(MemberAccess(MemberAccess(self, attn), forward), [x])`.
When compiling this, the compiler must:
1. Compile `self.attn` → I64 value (pointer to Attention model)
2. Know that `attn`'s type is `Model { name: "Attention", ... }`
3. Look up `__nsl_model_Attention_forward` in function registry
4. Call it with `[attn_ptr, x_val]`

This requires tracking the **type** of intermediate expressions during codegen. Currently,
`compile_expr` returns just a `Value`. We need to either:
- **Option A:** Return `(Value, Type)` from `compile_expr` — large refactor, many callers
- **Option B:** Maintain a type lookup table for expressions — adds complexity
- **Option C:** When compiling a MemberAccess on a model-typed variable, look up the field's
  type from the StructLayout + semantic type info to determine the sub-model's name

**Recommended: Option C.** It's the most targeted change. When we encounter
`MemberAccess(MemberAccess(self, attn), forward)`, we:
1. Recognize `self` is a Model → look up `attn` field → find it's `Type::Model("Attention")`
2. Now we know the inner MemberAccess produces an `Attention` pointer
3. Resolve `forward` as `__nsl_model_Attention_forward`

This requires storing the semantic type of each model field alongside the StructLayout, or
looking it up from the type checker's model registry. A simple `HashMap<String, HashMap<String, String>>`
mapping `model_name -> field_name -> field_model_type_name` suffices.

**Population timing:** This map is built during `collect_models()`, which already iterates
over all model fields. For each field whose type annotation resolves to a known model name,
record the mapping. The map is then available during expression compilation.

### 2.6 Memory Management

Sub-model pointers are heap-allocated by their constructors. When the parent model is freed,
the sub-model pointers must also be freed. Currently, model memory is never explicitly freed
(leaked). This is a known limitation that affects all models, not just nested ones. A proper
model destructor is a future improvement but not required for M18a correctness.

---

## Part 3: Fixed-Size Model Arrays

### 3.1 Syntax

```nsl
model Transformer(dim: int):
    layers: [TransformerBlock; 12]

    fn forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer.forward(x)
        return x
```

The `12` must be an **integer literal** — not a constructor parameter, not a variable, not an
expression. This keeps AOT struct sizing trivial: the compiler reads the literal directly from
the AST during layout computation.

### 3.2 Parser Changes

Add a new type expression variant for fixed-size model arrays:

```rust
// In nsl-ast/src/types.rs or wherever TypeExpr is defined
TypeExpr::FixedArray {
    element_type: Box<TypeExpr>,  // e.g., Named("TransformerBlock")
    size: i64,                     // The integer literal
    span: Span,
}
```

Parsing rule: when we see `[` in a type annotation position:
- Parse the element type
- Expect `;`
- Parse integer literal
- Expect `]`

This is unambiguous: `[TransformerBlock; 12]` can only appear as a type annotation.

### 3.3 Semantic Checker Changes

- Validate that the element type is a known model type
- Validate that the size is a positive integer literal
- Construct a new type: `Type::FixedModelArray { element_model: Symbol, size: i64 }`
- Store the array info alongside the model's field types

### 3.4 Layout Computation

In `collect_models()`, when encountering a FixedArray field:
1. Look up the element model's StructLayout (must be already collected — enforce ordering)
2. Each element is stored as a pointer (I64, 8 bytes)
3. Total array size = `8 * array_size` bytes (array of pointers)
4. Array field offset is aligned to 8 bytes

So a `layers: [TransformerBlock; 12]` field occupies 96 bytes (12 x 8-byte pointers).

### 3.5 Constructor Codegen

For a FixedArray field, the constructor must:
1. Allocate space for the array of pointers (already part of the parent struct allocation)
2. Loop (in generated Cranelift IR) to call the element constructor N times
3. Store each returned pointer at `base_offset + i * 8`

The element constructor arguments need to be available. Two approaches:
- **Approach A:** The array elements share the same constructor args as the parent model's
  params. E.g., `TransformerBlock` takes `dim`, and `dim` is a param of `Transformer`.
- **Approach B:** The field init expression specifies the constructor call explicitly:
  `layers: [TransformerBlock; 12] = [TransformerBlock(dim, 4) for 12]`

**Recommended: Approach B** with a simplified syntax. The user writes:
```nsl
model Transformer(dim: int):
    layers: [TransformerBlock; 12] = TransformerBlock(dim, 4)
```
**Semantics:** When the field type is `FixedArray` and the init expression is a single
constructor call (not a list), the codegen interprets it as "call this constructor N times
in a loop, storing each result." The init expression is re-evaluated on each iteration —
it is NOT evaluated once and the result copied. Each element gets its own independent
model instance with its own weights.

This is simpler than a full comprehension and covers the common case where all layers
have identical hyperparameters.

### 3.6 Array Field Access

**`self.layers`** — loads the base pointer (start of the array region in the struct).
Actually, since the array is inline in the struct (12 consecutive I64 pointers), accessing
`self.layers` returns the address of the first element.

**`for layer in self.layers`** — the compiler knows the array size (from the type) and
the element type. It generates a counted loop:
```
for i in 0..12:
    layer_ptr = load I64 from (self + layers_offset + i * 8)
    // compile loop body with layer_ptr as the loop variable
```

**`self.layers[i]`** — index access. The compiler bounds-checks `i < 12` (optional, can skip
for M18a) and loads `self + layers_offset + i * 8`.

### 3.7 Codegen IR Pattern

```
// In constructor:
block_loop_header:
    counter = phi(0, incremented)
    cmp = icmp_ult counter, 12
    brif cmp, block_loop_body, block_loop_exit

block_loop_body:
    sub_model_ptr = call __nsl_model_TransformerBlock(dim_val, heads_val)
    offset = iadd layers_base_offset, imul(counter, 8)
    addr = iadd self_ptr, offset
    store sub_model_ptr at addr
    incremented = iadd counter, 1
    jump block_loop_header

block_loop_exit:
    // continue with remaining field inits
```

### 3.8 For-Loop Iteration

When the compiler sees `for layer in self.layers` and knows `self.layers` has type
`FixedModelArray { element_model: "TransformerBlock", size: 12 }`:

1. Compile `self.layers` → get base address of array in struct
2. Generate counted loop `for i in 0..12`
3. In each iteration: `layer_ptr = load(base + i * 8)`
4. Bind `layer` variable to `layer_ptr` with type `Model("TransformerBlock")`
5. Compile loop body with `layer` in scope

This is a standard loop — no unrolling required for correctness (but the option exists
for future optimization).

---

## Part 4: Stdlib Updates

### 4.1 Multi-Head Attention (rewrite attention.nsl)

**Important: Constructor params are NOT accessible in `forward` methods.** Constructor params
are compile-time-only values used during field initialization. They are NOT stored in the model
struct. Any value needed at forward time must be explicitly stored as a field.

Workaround: store hyperparameters as `Tensor` fields (scalar tensors wrapping the value).
This is how the existing `Attention` model already works — `dim` is passed as a constructor
param and used only in field init expressions.

For multi-head attention, `num_heads` and `head_dim` are needed at forward time for reshape.
Store them as scalar tensors:

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

        # Project to Q, K, V
        let q = x @ self.q_proj
        let k = x @ self.k_proj
        let v = x @ self.v_proj

        # Reshape for multi-head: [seq, dim] -> [num_heads, seq, head_dim]
        let q = q.reshape([seq_len, nh, hd]).transpose(0, 1)
        let k = k.reshape([seq_len, nh, hd]).transpose(0, 1)
        let v = v.reshape([seq_len, nh, hd]).transpose(0, 1)

        # Scaled dot-product attention with causal mask
        let scale = sqrt(full([1], float(hd)))
        let scores = (q @ k.transpose(-2, -1)) / scale

        # Additive causal mask: 0.0 for allowed, -1e9 for masked
        let mask = causal_mask(seq_len)
        scores = scores + mask

        let attn = softmax(scores, -1)
        let out = attn @ v

        # Reshape back: [num_heads, seq, head_dim] -> [seq, dim]
        let out = out.transpose(0, 1).reshape([seq_len, d])
        return out @ self.out_proj
```

Note: This is single-sequence (no batch dimension) for M18a. Batched attention comes later.
The scalar tensor workaround for hyperparams is clunky but avoids a major codegen refactor.
A future milestone can add auto-storage of constructor params as hidden struct fields.

### 4.2 Transformer Block (new: stdlib/nsl/nn/transformer.nsl)

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
        # Pre-norm architecture with residual connections
        let h = x + self.attn.forward(self.norm1.forward(x))
        return h + self.mlp.forward(self.norm2.forward(h))
```

### 4.3 Positional Embeddings (new: stdlib/nsl/nn/position.nsl)

Simple learned positional embeddings for M18a:

```nsl
model PositionalEmbedding(max_seq: int, dim: int):
    pe: Tensor = randn([max_seq, dim])

    fn forward(self, x: Tensor) -> Tensor:
        # x has shape [seq_len, dim] where seq_len <= max_seq
        # Slice pe to match: pe[:seq_len, :] via slice(dim=0, start=0, end=seq_len)
        let seq_len = x.shape(0)
        let pos = self.pe.slice(0, 0, seq_len)
        return x + pos
```

Note: Uses `nsl_tensor_slice` (already implemented) to handle `seq_len < max_seq`.
Sinusoidal embeddings can be added later as a stdlib function.

### 4.4 Transformer Model (new: stdlib/nsl/nn/transformer.nsl, same file as 4.2)

```nsl
model Transformer(vocab_size: int, dim: int, num_heads: int, max_seq: int):
    embed: Embedding = Embedding(vocab_size, dim)
    pos_embed: PositionalEmbedding = PositionalEmbedding(max_seq, dim)
    layers: [TransformerBlock; 6] = TransformerBlock(dim, num_heads)
    norm: LayerNorm = LayerNorm(dim)
    lm_head: Linear = Linear(dim, vocab_size)

    fn forward(self, x: Tensor) -> Tensor:
        let h = self.embed.forward(x)
        h = self.pos_embed.forward(h)
        for layer in self.layers:
            h = layer.forward(h)
        h = self.norm.forward(h)
        return self.lm_head.forward(h)
```

---

## Part 5: End-to-End Test

### 5.1 Test Program (examples/m18_transformer.nsl)

```nsl
from nsl.nn.layers import Linear, Embedding, MLP
from nsl.nn.norms import LayerNorm
from nsl.nn.attention import Attention
from nsl.nn.position import PositionalEmbedding
from nsl.nn.losses import cross_entropy

# Define transformer components (inline for test — stdlib versions in stdlib/nsl/nn/)
model TransformerBlock(dim: int, num_heads: int):
    norm1: LayerNorm = LayerNorm(dim)
    attn: Attention = Attention(dim, num_heads)
    norm2: LayerNorm = LayerNorm(dim)
    mlp: MLP = MLP(dim, dim * 4, dim)

    fn forward(self, x: Tensor) -> Tensor:
        let h = x + self.attn.forward(self.norm1.forward(x))
        return h + self.mlp.forward(self.norm2.forward(h))

model TinyTransformer(vocab_size: int, dim: int, num_heads: int, max_seq: int):
    embed: Embedding = Embedding(vocab_size, dim)
    pos_embed: PositionalEmbedding = PositionalEmbedding(max_seq, dim)
    layers: [TransformerBlock; 2] = TransformerBlock(dim, num_heads)
    norm: LayerNorm = LayerNorm(dim)
    head: Linear = Linear(dim, vocab_size)

    fn forward(self, x: Tensor) -> Tensor:
        let h = self.embed.forward(x)
        h = self.pos_embed.forward(h)
        for layer in self.layers:
            h = layer.forward(h)
        h = self.norm.forward(h)
        return self.head.forward(h)

# Create model with tiny dims for testing
# Token IDs are represented as float tensors (embedding_lookup truncates to int internally)
let m = TinyTransformer(100, 32, 4, 64)

# Toy sequence: predict next token (8 tokens)
let input = full([8], 1.0)     # 8 tokens, all token-id 1
let target = full([8], 2.0)    # target: all token-id 2

# Forward pass test
let logits = m.forward(input)
print(logits)
print("Forward pass: PASS")

# Training test
train(model=m, epochs=3):
    optimizer: Adam(lr=0.001)
    step(batch):
        let pred = m.forward(input)
        let loss = cross_entropy(pred, target)
    callbacks:
        on_step(step, loss):
            print(loss)

print("Training: PASS")
print("m18_transformer: ALL PASS")
```

---

## Part 6: Risk Analysis

| Risk | Mitigation |
|------|------------|
| Chained member access (`self.attn.forward(x)`) requires type tracking through codegen | Option C (field-to-model-name lookup table) keeps changes minimal |
| Fixed-size arrays add loop codegen to constructors | Pattern already exists in for-loop compilation; reuse |
| `expand` copies data (no stride views) | Acceptable for M18a; note as future optimization |
| Nested model memory leaks (no destructors) | Same limitation as all models; not a regression |
| Autodiff for new ops | `unsqueeze`/`stack` backward are simple reshape/split; `expand` backward uses existing `sum_dim` |
| `causal_mask` on GPU | Generate on CPU, user calls `.to(cuda)`; avoids needing GPU kernel for mask generation |

---

## Part 7: Files Modified

| File | Change |
|------|--------|
| `crates/nsl-runtime/src/tensor.rs` | Add `nsl_tensor_unsqueeze`, `nsl_tensor_expand`, `nsl_tensor_stack`, `nsl_tensor_select`, `nsl_tensor_causal_mask` |
| `crates/nsl-runtime/src/autodiff.rs` | Add `TapeOp::Unsqueeze`, `TapeOp::Expand`, `TapeOp::Stack` + backward passes |
| `crates/nsl-ast/src/types.rs` or `decl.rs` | Add `TypeExpr::FixedArray { element_type, size }` |
| `crates/nsl-parser/src/decl.rs` | Parse `[Type; N]` in type annotation position |
| `crates/nsl-semantic/src/types.rs` | Add `Type::FixedModelArray { element_model, size }` |
| `crates/nsl-semantic/src/checker.rs` | Allow Model-typed fields; validate FixedArray fields |
| `crates/nsl-codegen/src/builtins.rs` | Register 5 new FFI functions |
| `crates/nsl-codegen/src/compiler.rs` | Handle FixedArray in `collect_models`, constructor loop codegen |
| `crates/nsl-codegen/src/expr.rs` | Chained model field access; `unsqueeze`/`expand` as methods; array indexing; for-in-array |
| `crates/nsl-codegen/src/stmt.rs` | `for layer in self.layers` compilation for FixedModelArray |
| `stdlib/nsl/nn/attention.nsl` | Rewrite with multi-head + causal masking |
| `stdlib/nsl/nn/transformer.nsl` | New: TransformerBlock, Transformer models |
| `stdlib/nsl/nn/position.nsl` | New: PositionalEmbedding |
| `examples/m18_transformer.nsl` | End-to-end test program |

---

## Part 8: Build Order

The implementation should proceed in this dependency order:

1. **Tensor ops** (runtime-only, no compiler changes) — can test via existing function call mechanism
2. **Autodiff for new ops** — extends tape, testable immediately
3. **Nested model fields** — semantic + codegen changes for model-typed fields
4. **Fixed-size model arrays** — parser + semantic + codegen for `[Type; N]`
5. **For-loop over model arrays** — codegen for iteration
6. **Stdlib updates** — attention, transformer block, position embeddings
7. **End-to-end test** — transformer forward + train
