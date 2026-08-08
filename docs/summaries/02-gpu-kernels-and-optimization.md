# NeuralScript: GPU Kernels & Optimization System

## Overview

NSL compiles GPU kernels at build time (not runtime), generating PTX for NVIDIA GPUs with partial support for AMDGPU, Metal, and WGSL backends. The optimization system includes multi-level operator fusion, memory planning, cost modeling, and build-time autotuning.

---

## GPU Kernel System

### Kernel Keyword
NSL provides a `kernel` keyword for writing custom GPU operations:
```
kernel vector_add(a: Tensor<[N], f32>, b: Tensor<[N], f32>) -> Tensor<[N], f32>:
    let tid = thread_idx_x()
    let idx = block_idx_x() * block_dim_x() + tid
    if idx < N:
        out[idx] = a[idx] + b[idx]
```

Kernel blocks compile to PTX at build time via the `KernelCompiler` in `nsl-codegen/src/kernel.rs`.

### CUDA Integration (M17)
- **Backend**: cudarc 0.19 with dynamic linking
- **Context**: `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` (CUDA 13.x removed cuCtxCreate)
- **Memory**: Unified memory via `cuMemAllocManaged` for zero-copy host/device access
- **Thread safety**: Thread-local CUDA contexts (must call `cuCtxSetCurrent` per-thread)
- **Dtype handling**: CPU = f64 (dtype=0), GPU = f32 (dtype=1), automatic conversion in `.to(device)`
- **PTX ISA 7.0**: Uses `mul.lo.u32` + `add.u32` (not `mad.lo.u32` which causes INVALID_PTX)

### Built-in GPU Kernels (15)
The runtime includes pre-compiled PTX kernels for:
- Arithmetic: add, sub, mul, div (element-wise, with broadcasting)
- Unary: neg, relu, exp, log, sqrt, abs, sign, sigmoid, tanh
- Scalar operations: scalar-tensor arithmetic
- Matrix multiplication: tiled matmul

### Multi-Backend Codegen (M47)

| Backend | File | Status | Target |
|---------|------|--------|--------|
| PTX (NVIDIA) | `backend_ptx.rs` | Production | sm_70+ (Volta, Ampere, Hopper) |
| AMDGPU | `backend_amdgpu.rs` | Framework | GFX90A (MI250X) |
| Metal | `backend_metal.rs` | Framework | Apple Silicon |
| WGSL | `backend_wgsl.rs` | Framework | WebGPU |

The `gpu_target.rs` abstraction and `kernel_ir.rs` intermediate representation (KIR) allow the same kernel logic to lower to different ISAs. In practice, only PTX is fully operational.

### GPU Architecture Specs Database
`gpu_specs.rs` contains hardware specs for: A100, H100, RTX-4090, Orin, and others. Used by the cost model and autotuner to make hardware-aware decisions.

---

## FlashAttention-2 (M27)

Production-quality implementation in `flash_attention.rs` (~2,600 lines):

- **Tiled attention** with online softmax (numerically stable)
- **Hopper-specific path** (sm_90+): runtime dispatch and codegen exist for wgmma-based kernels, but the generated TMA/wgmma path still contains placeholder pieces and simplified load plumbing rather than a fully finished production Hopper kernel
- **Ampere mma.sync** (sm_80): `mma.sync.aligned.m16n8k16` fallback for A100-class GPUs
- **Paged KV-cache** integration (block table lookup during attention)
- **RoPE fusion** (half-split and adjacent rotary embedding applied in-register)
- **GQA support** (grouped query attention with head mapping)
- **Tree-structured causal mask** (for speculative decoding, M33)
- **21+ configurable variants** parameterized by head dim, block sizes, causal/non-causal
- **Register pressure optimization**: processes one m-tile at a time
- **Logsumexp saving** for correct backward pass gradient computation
- **Fallback**: scalar FMA path for GPUs without tensor cores

Activated via `@flash_attention` decorator on attention functions.

---

## Operator Fusion System

NSL implements a multi-level fusion system across four modules:

### Level 1: Elementwise Fusion (M26)
**File**: `fusion.rs` (697 lines)

Fuses chains of element-wise operations into single GPU kernels:
```
# Before fusion: 3 kernel launches
let y = relu(x * w + b)

# After fusion: 1 kernel launch
# Synthesized PTX: load x, w, b → multiply → add → relu → store
```

- Lexical auto-detection of fusible chains
- Explicit `@fuse` decorator for manual control
- Synthesizes PTX with correct ISA (log2→ln conversion, ex2 adjustment)
- Supported ops: add, mul, relu, gelu, sigmoid, tanh, exp, log, sqrt, neg, silu

### Levels 2-4: Epilogue / reduction / graph fusion (M31) — REMOVED

`epilogue_fusion.rs`, `reduction_fusion.rs` and `fusion_graph.rs` described
matmul-epilogue fusion, map-into-reduction fusion, and DAG-level fusion
planning. All three were deleted (see the changelog): a `FusionGraph` was
never constructed outside `#[cfg(test)]`, so none of the three passes was
reachable from any compilation. `epilogue_fusion`'s PTX synthesiser also
emitted an un-indexed store against an indexed load — every thread would have
written element 0 — which nothing noticed because the kernel had never been
assembled, let alone launched.

What actually ships is Level 1 above (elementwise fusion, which does run) plus
the purpose-built fused emitters: fused linear-CE, fused RMSNorm backward,
fused AdamW, the CSHA fused attention kernels, and the cuBLAS `beta = 1`
weight-gradient accumulation (`--fuse-wgrad-accum`). Those are separate
emitters reached from `stmt.rs` and `wengert_lower.rs`, not from a general
fusion-graph framework.

`--fuse-wgrad-accum` has exactly one reachable path: the FASE-**Deferred**
`on_param_grad` hook, which requires `grad_accumulation >= 2` in the train
block. Below that FASE is `Passthrough` for every optimizer, no hook exists,
zero chains fuse — and until 2026-08-07 nothing said so, because the
`[wgrad-fusion] N chain(s) fused` counter was gated on the same hook.

Every train block that cannot reach the hook is now named on stderr —
`[wgrad-fusion] declined: train block #N — <reason>` — including one on the
`@pipeline` train path, which returns before the ordinary admission and used to
be the one place the flag stayed completely silent. The *refusal* is
compile-scoped, not per block: a typed `--fuse-wgrad-accum` fails the build only
when NO train block in the program could fuse, so a program that accumulates in
one block and not another still builds (and still gets the note). A
bundle-filled flag — `--pretrain-optimized` sets it on programs that never asked
— warns instead of failing in the same state.

---

## Memory Planning (M36)

**File**: `memory_planner.rs`

Compile-time tensor memory allocation:

1. **Liveness analysis**: Determines birth/death program points for each tensor allocation
2. **Interference graph**: Builds adjacency list of tensors whose lifetimes overlap
3. **Size classification**: Static (known at compile time), Bounded (max known), Dynamic
4. **Slab allocation**: Assigns non-overlapping tensors to shared memory slabs (reduces peak memory)

This is a compile-time-only feature — no runtime overhead. The planner produces allocation directives that the codegen uses to emit memory reuse code.

---

## Cost Model / Roofline Analysis (M37)

**File**: `cost_model.rs`

Per-operation FLOP and byte transfer calculations:

| Operation | FLOP Formula | Bytes Formula |
|-----------|-------------|---------------|
| Matmul [M,K]@[K,N] | 2·M·K·N | (M·K + K·N + M·N)·sizeof |
| Softmax [B,S] | 7·B·S | 2·B·S·sizeof |
| LayerNorm [B,S,D] | 8·B·S·D | (2·B·S·D + 2·D)·sizeof read, B·S·D·sizeof write |
| RMSNorm | 5·B·S·D | (B·S·D + D)·sizeof read, B·S·D·sizeof write |
| Elementwise | B·S·D | 2·B·S·D·sizeof |
| FlashAttention | 4·B·H·S²·D | 3·B·H·S·D·sizeof read, B·H·S·D·sizeof write |
| Reduction | elements | elements·sizeof read, (elements / reduced_dim)·sizeof write |

Classifies operations as compute-bound, memory-bound, or balanced based on arithmetic intensity (FLOP/byte ratio vs. hardware peak).

Activated via `nsl check --perf file.nsl`.

---

## Build-Time Autotuning (M26)

**File**: `autotune.rs`

The `@autotune` decorator enables build-time kernel parameter search:
```
@autotune(block_size=[64, 128, 256], tile_size=[16, 32])
kernel my_kernel(...):
    ...
```

- Generates Cartesian product of parameter variants
- Benchmarks each variant on the target GPU during compilation
- Selects the fastest configuration
- Caches results for subsequent builds

Currently: framework with median-value fallback. Full GPU benchmarking (cuEvent timing, cache persistence) is planned.

---

## FP8 Compute (M35)

**File**: `fp8.rs` (codegen + runtime)

Production-quality FP8 support targeting H100 (sm_90):

- **MMA instruction**: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`
- **Formats**: E4M3 (forward/inference), E5M2 (backward/training gradients)
- **Calibration**: Running-max EMA for per-tensor scale factors
- **Dispatch**: sm_90 → MMA kernel, older GPUs → dequantize-to-f32 fallback
- **Quantization/dequantization**: Round-to-nearest with saturation

Activated via `@fp8_compute` decorator.

**Blackwell support status**: MXFP8 per-block scaling (E8M0 scale per 32 elements) and NVFP4 (E2M1 with Hadamard preprocessing) have runtime quantizers and decorator/config parsing in tree, but end-to-end lowering and hardware-specific integration are still follow-up work.

---

## Quantization System (M16, M21-M22, M35)

### Compile-Time Quantization
```
quant static:
    model = MyModel()
    scheme = "per_channel"
    bits = 8
    calibration_data = load_data("cal.bin")
```

### Supported Schemes
| Scheme | Bits | Granularity | Implementation |
|--------|------|-------------|----------------|
| INT8 | 8 | per-tensor, per-channel, per-group | Full |
| INT4 | 4 | per-group | Full |
| FP8 E4M3 | 8 | per-tensor | Full (H100 MMA) |
| FP8 E5M2 | 8 | per-tensor | Full |
| AWQ | 4 | activation-aware groups | Full |
| GPTQ | 4/8 | column-wise | Full |

### QuantizedTensor Type
A distinct type in the type system — the compiler tracks quantized tensors separately from regular tensors, ensuring correct dequantization before incompatible operations.

### Custom Datatypes (M23)
```
datatype bfloat8:
    bits = 8
    exponent = 5
    mantissa = 2
    bias = 15
    pack_fn = ...
    unpack_fn = ...
```

The `datatype` block defines custom numeric formats with block-aware packing and PTX escape hatches for hardware-specific operations.

---

## Standalone Export (M24)

```bash
nsl build --standalone model.nsl -o inference_server
```

Produces a zero-dependency native binary with:
- Model weights embedded or as sidecar file
- No Python, no CUDA toolkit needed at runtime (CUDA driver only)
- Adaptive weight bundling (embed small weights, sidecar large ones)
- SHA-256 integrity checking for weights

---

## Weight-Aware Compilation (M52)

**File**: `weight_aware.rs` (1,364 lines)

The compiler loads `.safetensors` weight files at compile time and optimizes:

1. **Constant folding**: Pre-computes matmul/add/relu with known weight operands
2. **Dead weight elimination**: Removes near-zero weights below threshold
3. **Sparsity analysis**: Detects and reports per-tensor sparsity (CSR layout)
4. **Near-identity detection**: Identifies matrices close to identity (potential no-ops)
5. **Integrity**: SHA-256 checksums for weight verification

```bash
nsl build --weights model.safetensors --standalone model.nsl
nsl check --weight-analysis model.nsl --weights model.safetensors
```

This is NSL's flagship feature — no other system compiles model weights into the binary with constant folding. It produces per-checkpoint bespoke binaries.

---

## Kernel Profiling

The runtime includes kernel-level profiling (`kernel_profiler.rs`) and Chrome tracing output for visualization:

```bash
nsl run --trace model.nsl   # generates Chrome trace JSON
```

Traces capture kernel launch times, memory transfers, and tensor lifecycle events.
