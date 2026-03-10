# Section 2 — Tensor Type System

## Design Rationale

The single biggest source of bugs in PyTorch code is shape mismatches discovered at runtime.
NSL makes tensors a first-class type with compile-time shape tracking, named dimensions, and
device awareness. The type system can statically verify that a matmul's inner dimensions match,
that a reshape preserves element count, and that tensors on different devices aren't mixed
without explicit transfer. This catches 80%+ of ML bugs before the code ever runs.

## Tensor Type Grammar

```ebnf
tensor_type     ::= 'Tensor' '<' shape_spec ',' dtype_spec (',' device_spec)? '>'
shape_spec      ::= '[' dim_entry (',' dim_entry)* ']'
dim_entry       ::= concrete_dim | symbolic_dim | named_dim | wildcard_dim
concrete_dim    ::= INTEGER                          # e.g., 768
symbolic_dim    ::= IDENT                            # e.g., batch (resolved at compile/runtime)
named_dim       ::= IDENT '=' (STRING | INTEGER)     # e.g., heads="H" or seq=2048
wildcard_dim    ::= '_'                              # any dimension (unchecked)

dtype_spec      ::= 'fp64' | 'fp32' | 'fp16' | 'bf16'
                  | 'fp8_e4m3' | 'fp8_e5m2'
                  | 'int64' | 'int32' | 'int16' | 'int8' | 'int4'
                  | 'uint8' | 'bool'

device_spec     ::= 'cpu' | 'cuda' ('(' INTEGER ')')? | 'metal' | 'rocm'
                  | 'npu' '<' npu_target '>'
npu_target      ::= IDENT                           # e.g., QuadricChimera, EdgeTPU

# Derived tensor types
grad_tensor     ::= 'GradTensor' '<' shape_spec ',' dtype_spec '>'    # requires_grad=true
param_tensor    ::= 'Param' '<' shape_spec ',' dtype_spec '>'         # model parameter
buffer_tensor   ::= 'Buffer' '<' shape_spec ',' dtype_spec '>'        # non-trainable state
sparse_tensor   ::= 'Sparse' '<' shape_spec ',' dtype_spec ',' format '>'
format          ::= 'coo' | 'csr' | 'csc' | 'bsr'
```

## Compile-Time Shape Rules

The compiler maintains a **shape algebra** that tracks tensor shapes through operations:

```
# Shape propagation rules (compiler-internal)
matmul(A: [M, K], B: [K, N])           -> [M, N]        # inner dims must match
add(A: [M, N], B: [M, N])              -> [M, N]        # shapes must match (or broadcast)
broadcast(A: [M, 1], B: [1, N])        -> [M, N]        # explicit broadcast
reshape(A: [M, N], shape: [M*N])       -> [M*N]         # element count preserved
transpose(A: [M, N])                   -> [N, M]
cat([A: [M, K], B: [N, K]], dim=0)     -> [M+N, K]      # concat along dim 0
unsqueeze(A: [M, N], dim=0)            -> [1, M, N]
```

When a symbolic dimension is used (e.g., `batch`), the compiler tracks it as a
**type variable** — it can unify with any concrete value but must be consistent
within a scope. If `batch` appears in two tensors in the same function, they must
have the same batch dimension.

Wildcard `_` opts out of checking for that dimension — useful for dynamic shapes
or when you truly don't care.

## Named Dimensions

Named dimensions solve the "dim=1 vs dim=-1" confusion:

```nsl
# Instead of remembering dimension indices, name them
let attn: Tensor<[batch="B", heads="H", seq="S", d_k=64], fp32, cuda>

# Operations reference names instead of indices
let scores = attn.sum(dim="heads")           # clear intent
let transposed = attn.transpose("seq", "d_k")  # swap named dims

# The compiler checks that named dims are compatible
fn multi_head_attention(
    q: Tensor<[batch, heads, seq, d_k], fp32>,
    k: Tensor<[batch, heads, seq, d_k], fp32>,   # d_k must match q's d_k
    v: Tensor<[batch, heads, seq, d_v], fp32>
) -> Tensor<[batch, heads, seq, d_v], fp32>:
    let scores = q @ k.transpose("seq", "d_k")   # [batch, heads, seq, seq]
    let weights = softmax(scores, dim="d_k")
    return weights @ v                             # [batch, heads, seq, d_v]
```

## Autobroadcast Rules

Broadcasting is explicit in NSL — the compiler will not silently broadcast. You must
annotate broadcast intent:

```nsl
# COMPILE ERROR: shapes don't match
# let c = a + b   # where a: [3, 4] and b: [4]

# Explicit broadcast
let c = a + b.broadcast_to([3, 4])          # OK: explicit
let c = a + b.unsqueeze(0).expand([3, 4])   # OK: manual

# Or use the broadcast annotation on the operation
let c = (a + b) @broadcast                  # OK: compiler applies broadcast rules

# Compile-time broadcast validation
# The compiler checks that broadcast is valid per NumPy broadcast rules:
# - Dimensions are compared from right to left
# - Each dimension must be equal, or one of them must be 1
```

## 8 Tensor Declaration Examples

```nsl
# Example 1: Simple 2D tensor on CPU (default device)
let weights: Tensor<[768, 3072], fp32> = zeros([768, 3072])

# Example 2: Batched 3D tensor with symbolic batch dimension on GPU
let hidden: Tensor<[batch, seq, 768], bf16, cuda> = zeros([batch, seq, 768], dtype=bf16, device=cuda)

# Example 3: Named dimensions for attention
let qkv: Tensor<[batch="B", heads=12, seq="S", d_model=64], fp16, cuda>

# Example 4: Quantized weight tensor (int4 for inference)
let w_quant: Tensor<[4096, 4096], int4, cuda> = quantize(weights, scheme=int4)

# Example 5: FP8 activation tensor for training
let activations: Tensor<[batch, seq, 768], fp8_e4m3, cuda>

# Example 6: Sparse tensor in CSR format
let sparse_attn: Sparse<[seq, seq], fp32, csr> = to_sparse(attention_mask)

# Example 7: Model parameter (automatically tracked for gradients)
let wq: Param<[d_model, d_model], bf16> = init.kaiming([d_model, d_model])

# Example 8: Buffer (non-trainable, e.g., positional encoding cache)
let pos_cache: Buffer<[1, MAX_SEQ, d_model], fp32> = precompute_rotary(MAX_SEQ, d_model)
```

## Shape Arithmetic & Compile-Time Validation

```nsl
# The compiler validates shapes at every operation:

fn linear_forward(x: Tensor<[batch, seq, 768], bf16>,
                  w: Tensor<[768, 3072], bf16>,
                  b: Tensor<[3072], bf16>) -> Tensor<[batch, seq, 3072], bf16>:
    # Compiler verifies: [batch, seq, 768] @ [768, 3072] = [batch, seq, 3072] ✓
    return (x @ w) + b @broadcast

# Shape errors are compile-time errors:
fn bad_matmul(a: Tensor<[32, 64], fp32>, b: Tensor<[128, 64], fp32>):
    # return a @ b
    # ^ COMPILE ERROR: matmul inner dimensions don't match:
    #   a has shape [32, 64], b has shape [128, 64]
    #   Expected b to have shape [64, _]
    return a @ b.transpose(0, 1)  # OK: [32, 64] @ [64, 128] = [32, 128]
```

## Type Coercion Rules

```nsl
# Implicit promotion (safe widening only):
# int4 -> int8 -> int16 -> int32 -> int64
# fp8 -> fp16 -> fp32 -> fp64
# bf16 -> fp32 -> fp64

# Narrowing requires explicit cast:
let x: Tensor<[10], fp32> = rand([10])
let y: Tensor<[10], fp16> = x.to(fp16)        # explicit cast required
# let z: Tensor<[10], fp16> = x               # COMPILE ERROR: narrowing without cast

# Device transfer is always explicit:
let cpu_tensor: Tensor<[10], fp32, cpu> = rand([10])
let gpu_tensor = cpu_tensor.to(cuda)           # explicit device transfer
# let bad = cpu_tensor + gpu_tensor            # COMPILE ERROR: device mismatch
```

## Design Tensions & Tradeoffs

1. **Compile-time shapes vs Dynamic shapes**: Some models (e.g., NLP with variable sequence
   lengths) genuinely need dynamic shapes. NSL handles this with symbolic dimensions that
   are checked for consistency but not for concrete values. The `_` wildcard opts out entirely.
   This is a spectrum from full safety to full flexibility.

2. **Explicit broadcast vs Convenience**: Requiring explicit broadcasts adds verbosity but
   eliminates an entire class of silent bugs (e.g., accidentally broadcasting a [1, 768]
   bias across a [32, 768, 768] attention matrix). The `@broadcast` annotation is a
   compromise — one token to opt into NumPy-style broadcasting.

3. **Named dimensions overhead**: Storing dimension names at compile time is free. At runtime,
   named dims are erased — they're purely a compile-time construct. No runtime overhead.

4. **Param vs Tensor**: Separating `Param` from `Tensor` adds a type, but it makes gradient
   tracking explicit in the type system. You can never accidentally forget `requires_grad` —
   if it's a `Param`, it's differentiable. If it's a `Buffer`, it's not.
