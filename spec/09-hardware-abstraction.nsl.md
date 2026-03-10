# Section 9 — Hardware Abstraction Layer (HAL)

## Design Rationale

ML workloads run on wildly diverse hardware: NVIDIA GPUs (CUDA), AMD GPUs (ROCm), Apple
Silicon (Metal), and edge NPUs (Quadric Chimera, Google EdgeTPU). PyTorch abstracts hardware
through backends, but custom kernels still require CUDA C, Triton, or Metal shaders — each a
separate language. NSL provides a unified `kernel` keyword for writing hardware-specific
operations inline, with the compiler targeting the appropriate backend. The `device` type
system prevents accidental cross-device operations.

## Device Type System

```ebnf
device_type     ::= 'CPU' | 'CUDA' ('(' INT ')')? | 'Metal' | 'ROCm' ('(' INT ')')?
                   | 'NPU' '<' npu_target '>'
npu_target      ::= 'QuadricChimera' | 'EdgeTPU' | 'QualcommHexagon' | IDENT

# Device is a first-class type — tensors carry their device
# The compiler prevents operations between tensors on different devices
device_constraint ::= 'where' 'device' '==' device_type
```

## Kernel Block Grammar

```ebnf
kernel_def      ::= ('@autotune' '(' autotune_config ')')? ('@fuse')?
                     'kernel' IDENT '(' param_list ')' ('->' type)? ':' INDENT kernel_body DEDENT
kernel_body     ::= (kernel_stmt NEWLINE)*
kernel_stmt     ::= 'grid' '=' grid_spec
                   | 'block' '=' block_spec
                   | 'shared' IDENT ':' type '=' expression
                   | statement  # standard NSL with hardware intrinsics

grid_spec       ::= '(' expression (',' expression)* ')'
block_spec      ::= '(' expression (',' expression)* ')'

autotune_config ::= config_entry (',' config_entry)*

# Hardware intrinsic functions (available inside kernel blocks)
intrinsic       ::= 'thread_id' '(' INT ')'      # thread index in dimension
                   | 'block_id' '(' INT ')'       # block index in dimension
                   | 'warp_id' '()'               # warp/wavefront index
                   | 'sync_threads' '()'           # barrier synchronization
                   | 'atomic_add' '(' expr ',' expr ')'
                   | 'shared_load' '(' expr ')'
                   | 'shared_store' '(' expr ',' expr ')'
```

## 3 Hardware Abstraction Examples

### Example 1: Fused Softmax Kernel

```nsl
## A custom fused softmax kernel that operates entirely in shared memory.
## This replaces 3 separate kernels (max, subtract+exp, sum+divide) with one,
## eliminating global memory round-trips.

@autotune(
    block_sizes=[32, 64, 128, 256],    # try these block sizes
    num_warps=[2, 4, 8],                # try these warp counts
    metric=throughput                    # optimize for throughput
)
kernel fused_softmax(
    input: Tensor<[batch, seq], fp32>,
    output: Tensor<[batch, seq], fp32>
):
    # Grid: one block per row (batch element)
    grid = (batch,)
    block = (seq,)    # autotune may override this

    let tid = thread_id(0)
    let bid = block_id(0)

    # Load row into shared memory
    shared row: [seq] fp32
    row[tid] = input[bid, tid]
    sync_threads()

    # Step 1: Find max (parallel reduction)
    shared max_val: fp32 = -inf
    atomic_max(max_val, row[tid])
    sync_threads()

    # Step 2: Subtract max and exponentiate
    row[tid] = (row[tid] - max_val).exp()
    sync_threads()

    # Step 3: Sum (parallel reduction)
    shared sum_val: fp32 = 0.0
    atomic_add(sum_val, row[tid])
    sync_threads()

    # Step 4: Normalize
    output[bid, tid] = row[tid] / sum_val

# Usage — the kernel is called like any function
let probs = fused_softmax(logits)    # compiler dispatches to the optimized kernel
```

### Example 2: Matrix Multiply with Operator Fusion

```nsl
## Demonstrates @fuse annotation — the compiler fuses multiple operations
## into a single kernel to avoid intermediate memory allocation.

# Without fusion: 3 kernels, 2 intermediate tensors
fn unfused_gelu_mlp(x: Tensor, w1: Tensor, w2: Tensor, b1: Tensor) -> Tensor:
    let h = x @ w1 + b1     # kernel 1: matmul, kernel 2: add
    let h = gelu(h)          # kernel 3: gelu activation
    return h @ w2            # kernel 4: matmul

# With @fuse: compiler merges compatible operations
@fuse
fn fused_gelu_mlp(x: Tensor, w1: Tensor, w2: Tensor, b1: Tensor) -> Tensor:
    let h = x @ w1 + b1     # fused into 1 kernel: matmul + bias
    let h = gelu(h)          # fused: gelu applied in-place after matmul
    return h @ w2            # separate kernel (matmul can't fuse across matmuls)

# Result: 2 kernels instead of 4, 0 intermediate tensors instead of 2

# Custom tiled matrix multiply targeting specific hardware
@autotune(
    tile_m=[32, 64, 128],
    tile_n=[32, 64, 128],
    tile_k=[16, 32],
    metric=throughput
)
kernel tiled_matmul(
    A: Tensor<[M, K], bf16, cuda>,
    B: Tensor<[K, N], bf16, cuda>,
    C: Tensor<[M, N], fp32, cuda>
):
    const TILE_M = autotune.tile_m
    const TILE_N = autotune.tile_n
    const TILE_K = autotune.tile_k

    grid = (M // TILE_M, N // TILE_N)
    block = (TILE_M, TILE_N)

    let row = block_id(0) * TILE_M + thread_id(0)
    let col = block_id(1) * TILE_N + thread_id(1)

    # Shared memory tiles
    shared A_tile: [TILE_M, TILE_K] bf16
    shared B_tile: [TILE_K, TILE_N] bf16

    let acc: fp32 = 0.0

    # Tile over K dimension
    for k_start in (0..K).step(TILE_K):
        # Cooperative load into shared memory
        A_tile[thread_id(0), thread_id(1)] = A[row, k_start + thread_id(1)]
        B_tile[thread_id(0), thread_id(1)] = B[k_start + thread_id(0), col]
        sync_threads()

        # Compute partial dot product
        for k in 0..TILE_K:
            acc += (A_tile[thread_id(0), k] * B_tile[k, thread_id(1)]).to_fp32()
        sync_threads()

    C[row, col] = acc
```

### Example 3: Edge NPU Kernel (Quadric Chimera)

```nsl
## Custom kernel targeting the Quadric Chimera GPNPU.
## The Chimera has INT4x8 MAC arrays, scratchpad memory, and a
## tile-based execution model. This kernel does quantized attention.

kernel chimera_int4_attention(
    q: Tensor<[seq, d_model], int8, npu<QuadricChimera>>,
    k: Tensor<[seq, d_model], int4, npu<QuadricChimera>>,
    v: Tensor<[seq, d_model], int4, npu<QuadricChimera>>,
    q_scale: Scale<fp16>,
    k_scale: Scale<fp16>,
    v_scale: Scale<fp16>,
    output: Tensor<[seq, d_model], int8, npu<QuadricChimera>>
):
    # Chimera-specific: use scratchpad memory instead of shared memory
    const TILE_SIZE = 64    # matches Chimera tile width

    grid = (seq // TILE_SIZE,)
    block = (TILE_SIZE,)

    let tile_id = block_id(0)

    # Load Q tile into scratchpad (Chimera's fast local memory)
    scratchpad q_tile: [TILE_SIZE, d_model] int8
    q_tile = q[tile_id * TILE_SIZE : (tile_id + 1) * TILE_SIZE, :]

    # Compute attention scores using INT4x8 MAC arrays
    # Q (int8) * K^T (int4) → int32 accumulator → scale to fp16 → softmax → int8
    scratchpad scores: [TILE_SIZE, seq] int32

    for k_tile_id in 0..(seq // TILE_SIZE):
        scratchpad k_tile: [TILE_SIZE, d_model] int4
        k_tile = k[k_tile_id * TILE_SIZE : (k_tile_id + 1) * TILE_SIZE, :]

        # MAC array: int8 * int4 → int32 accumulation
        # This maps directly to Chimera's hardware MAC units
        @mac_array(precision=int4x8)
        scores[:, k_tile_id * TILE_SIZE : (k_tile_id + 1) * TILE_SIZE] =
            q_tile @ k_tile.transpose()

    # Apply scales and softmax
    let scaled_scores = scores.to_fp16() * (q_scale * k_scale)
    let attn_weights = softmax_int8(scaled_scores)    # quantized softmax

    # Weighted sum of values
    scratchpad result: [TILE_SIZE, d_model] int32
    for v_tile_id in 0..(seq // TILE_SIZE):
        scratchpad v_tile: [TILE_SIZE, d_model] int4
        v_tile = v[v_tile_id * TILE_SIZE : (v_tile_id + 1) * TILE_SIZE, :]

        @mac_array(precision=int4x8)
        result += attn_weights[:, v_tile_id * TILE_SIZE : (v_tile_id + 1) * TILE_SIZE] @ v_tile

    # Write result with output quantization
    output[tile_id * TILE_SIZE : (tile_id + 1) * TILE_SIZE, :] =
        quantize_int8(result.to_fp16() * v_scale)
```

## Hardware Feature Queries

```nsl
# Query hardware capabilities at compile time or runtime
if device.cuda.compute_capability >= (8, 0):
    # SM 8.0+ (Ampere): use TF32 tensor cores
    let output = matmul(a, b, mode=tf32)
elif device.cuda.compute_capability >= (7, 0):
    # SM 7.0+ (Volta): use FP16 tensor cores
    let output = matmul(a, b, mode=fp16)
else:
    # Fallback: standard FP32
    let output = matmul(a, b, mode=fp32)

# Device introspection
print(f"device: {device.name}")                  # "NVIDIA A100 80GB"
print(f"memory: {device.memory_total_gb:.0f}GB")  # 80
print(f"compute: {device.compute_capability}")    # (8, 0)
print(f"SM count: {device.multiprocessor_count}") # 108
```

## Design Tensions & Tradeoffs

1. **Abstraction vs Performance**: A higher-level kernel language is easier to write but may
   not match the performance of hand-tuned CUDA C or Triton. NSL's `@autotune` mitigates this
   by searching over implementation parameters, and the compiler applies standard optimizations
   (vectorization, loop unrolling, register tiling).

2. **Unified kernel language vs Hardware-specific code**: The kernel syntax is the same across
   targets, but hardware-specific intrinsics (like Chimera's `@mac_array` or CUDA's `warp_shuffle`)
   are target-specific. Code using these intrinsics is not portable.

3. **When to write kernels**: Most users should never write kernels — NSL's built-in operations
   and `@fuse` handle 95% of cases. Kernels are for library authors and performance engineers
   who need the last 5% of performance.
