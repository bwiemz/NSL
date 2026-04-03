# NeuralScript: Analysis, Safety & Frontier Features

## Overview

NSL's compile-time analysis features represent its deepest competitive moat — these are capabilities that are structurally impossible in Python/PyTorch due to their requirement for static analysis of the full computation graph. This document covers type safety, effect systems, determinism, WCET proofs, ZK inference, and other frontier features.

---

## Compile-Time Analysis Suite

NSL provides multiple analysis passes accessible via CLI flags:

```bash
nsl check --perf file.nsl          # Roofline performance analysis
nsl check --nan-analysis file.nsl  # NaN/Inf risk detection
nsl check --deterministic file.nsl # Determinism verification
nsl check --wcet file.nsl          # Worst-case execution time
nsl check --weight-analysis file.nsl --weights model.safetensors  # Weight analysis
```

---

## Linear Types / Ownership System (M38)

**File**: `nsl-semantic/src/ownership.rs`

NSL's ownership system brings Rust-like move semantics to tensor programming:

- **Use-after-move detection**: Compiler catches attempts to use a tensor after it has been consumed
- **Immutable borrow syntax (`&T`)**: Functions can borrow tensors without consuming them. Auto-borrow at call sites (`T` auto-converts to `&T`).
- **Borrow safety rules**: Cannot consume while borrowed, no borrows in return types, no `&mut T` (unsupported), no nested `&&T`
- **Tape safety**: Borrowed tensors are transparent to the autodiff tape — the borrow passes through the same raw pointer as the owned tensor
- **`@shared` escape hatch**: Marks tensors as refcounted for shared ownership (e.g., model weights used in multiple forward passes)
- **Codegen integration**: Ownership decision tree (FreeAtConsumption, TapeHoldsReference, BorrowedNoAction), refcount elision for proven linear bindings

### Why This Matters
In Python/PyTorch, tensor aliasing bugs cause silent correctness errors:
```python
x = model(input)
y = x.view(...)   # y aliases x's memory
x += 1            # silently corrupts y
```

In NSL, the ownership checker prevents this at compile time.

### Current Status
- **FBIP**: Refcount-checked and compiler-proven in-place mutation — runtime `refcount == 1` fast path plus static reuse analysis and GPU in-place kernels are **implemented**

---

## Effect System (M51)

**File**: `nsl-semantic/src/effects.rs`

Tracks and controls side effects at the function level:

### Effect Categories
| Effect | Description | Example Operations |
|--------|-------------|-------------------|
| `IO` | File/network I/O | print, read_file, save_checkpoint |
| `RANDOM` | Non-deterministic randomness | rand, randn, dropout |
| `MUTATION` | In-place tensor modification | copy_data, in-place ops |
| `COMMUNICATION` | Inter-device/process communication | all_reduce, ring_send |

### Effect Annotations
```
@pure
fn relu(x: Tensor) -> Tensor:      # No side effects allowed
    return max(x, 0.0)

@deterministic
fn my_layer(x: Tensor) -> Tensor:  # No RANDOM effect allowed
    ...
```

### Effect Inference
The compiler infers effects by:
1. Cataloging built-in function effects (40+ operations classified)
2. Propagating effects through the call graph (union of callee effects)
3. Checking declared effect constraints against inferred effects

### Why This Matters
Effect tracking enables:
- **Reproducibility verification**: A function marked `@deterministic` that calls `rand()` is a compile error
- **Parallelism safety**: Functions without MUTATION/COMMUNICATION effects can be safely parallelized
- **Optimization**: Pure functions can be memoized, reordered, or eliminated if unused

---

## Shape Algebra Solver (M49)

**File**: `nsl-semantic/src/shape_algebra.rs`

A constraint-based solver for symbolic tensor dimensions:

### Constraint Types
```
# Concrete binding
let batch = 32

# Divisibility
assert batch divides total_elements

# Range bounds
assert 1 <= seq_len <= 4096

# Equality
assert hidden == heads * head_dim
```

### Solver Capabilities
- **Concrete evaluation**: Resolves known symbols to values
- **Expression simplification**: Reduces arithmetic expressions
- **Constraint propagation**: Derives new facts from existing constraints
- **Reshape validation**: Proves element count preservation across reshapes

### Example
```
fn reshape_safe(x: Tensor<[B, S*D], f32>) -> Tensor<[B, S, D], f32>:
    return x.reshape([B, S, D])
    # Compiler proves: B * (S*D) == B * S * D ✓
```

---

## NaN/Inf Risk Detection

**File**: `nsl-semantic/src/nan_analysis.rs`

Static analysis that identifies operations at risk of producing NaN or Inf:

- **Division by zero**: Detects `x / y` where `y` could be zero
- **Log of non-positive**: Flags `log(x)` where `x` might be ≤ 0
- **Exp overflow**: Warns about `exp(x)` with unbounded `x`
- **Softmax stability**: Checks for numerically unstable softmax implementations
- **Gradient explosion**: Identifies paths in the computation graph prone to gradient overflow

Activated via `nsl check --nan-analysis`.

---

## Determinism / Reproducibility (M46)

**Files**: `nsl-semantic/src/determinism.rs`, `nsl-codegen/src/deterministic_kernels.rs`, `nsl-runtime/src/deterministic_ops.rs`

### Semantic Analysis
- **DeterminismMode**: Off, FunctionLevel, Global
- **RNG tracking**: Classifies random sources as ExplicitSeed, Derived, or Implicit
- **Non-determinism categories**: GpuAtomic, ImplicitRng, AlgorithmSelection, External
- **Function-level marking**: `@deterministic` decorator with compile-time verification

### Kernel Variants
The codegen can select deterministic variants of GPU kernels:
- `DeterministicSortReduce`: Sort-based reduction (deterministic but slower)
- `DeterministicSortAccumulate`: Deterministic accumulation order
- `DeterministicCublas`: Force deterministic cuBLAS algorithms

### Graph Fingerprinting
Computes a hash of the computation graph for bitwise reproducibility verification across runs.

---

## Performance Budget Validation

**File**: `nsl-semantic/src/perf_budget.rs`

Annotate functions with FLOP/byte budgets and the compiler verifies compliance:
```
@budget(max_flops=1e12, max_bytes=1e9)
fn my_layer(x: Tensor<[B, S, D], f16>) -> Tensor<[B, S, D], f16>:
    ...
```

Uses the cost model (M37) to estimate FLOP and memory transfer counts, then checks against declared budgets.

---

## WCET Proofs (M53)

**File**: `nsl-codegen/src/wcet.rs`

Worst-Case Execution Time analysis for safety-critical inference (robotics, aerospace):

```
@real_time(max_latency_ms=10.0)
fn safety_critical_inference(sensor_data: Tensor) -> Tensor:
    ...

@wcet_budget(max_cycles=1000000)
fn motor_control(input: Tensor) -> Tensor:
    ...
```

### Analysis
1. Extracts timing decorators from functions
2. Computes per-operation WCET using the roofline cost model
3. Sums worst-case paths through the computation graph
4. Generates JSON certificates with safety margins
5. Reports pass/fail against declared budgets

### Target
DO-178C compliance reporting for aerospace applications — static proof that inference will complete within a guaranteed time bound.

### Requirements
- Zero dynamic memory allocation in the inference path
- Known instruction counts via the cost model
- No data-dependent branching (static computation graph)

---

## ZK Inference Circuits (M55)

**File**: `nsl-codegen/src/zk/` (3,097 lines)

Compiles model forward passes to zero-knowledge proof circuits:

### Purpose
Prove that a specific model produced a specific output for a specific input, without revealing the model weights (or input, depending on privacy mode).

### Privacy Modes
| Mode | Public | Private |
|------|--------|---------|
| `weight_private` | Input + Output | Weights |
| `input_private` | Output + Weights | Input |
| `full_private` | Output only | Weights + Input |
| `architecture_attestation` | Architecture | Everything else |

### Implementation
- **IR lowering**: NSL computation graph → PLONKish constraints
- **Field arithmetic**: Operations over finite fields (BN254, BLS12-381)
- **Lookup tables**: Non-linear activations (ReLU, GELU) via pre-computed lookup tables
- **Fixed-point arithmetic**: Floating-point → fixed-point conversion for field compatibility
- **Backend abstraction**: `ZkBackend` trait with Halo2 and Plonky3 backends
- **Solidity emitter**: Generates on-chain verifier contracts

### Why NSL Can Do This
ZK circuits require a static computation DAG with no data-dependent branching. Python's dynamic dispatch makes this impossible. NSL's AOT compilation with static graphs is a natural fit.

---

## Sparse Tensor Compilation (M50)

**Codegen**: `sparse.rs`
**Semantic**: `sparse.rs`
**Runtime**: `sparse.rs`

### Sparse Formats
| Format | Description | Best For |
|--------|-------------|----------|
| COO | Coordinate (row, col, value arrays) | Construction, conversion |
| CSR | Compressed Sparse Row | Row-major SpMV/SpMM |
| CSC | Compressed Sparse Column | Column-major operations |
| BSR | Block Sparse Row | Structured sparsity (2:4) |

### Implementation
- **Type system**: `Sparse<format, [shape], dtype>` as distinct type
- **`@layout` annotation**: Format-agnostic sparsity — users annotate tensors with `@layout("CSR")` and write standard dense equations; compiler auto-generates optimal sparse iteration
- **TACO-style merge lattices**: Intersection (for multiply) and union (for add) co-iteration over multiple sparse tensors
- **Level format composition**: Dense, Compressed, Singleton, CompressedNonUnique, Hashed — compose per-dimension to express CSR, CSC, COO, BCSR, and custom formats
- **Concrete index notation lowering**: Translates tensor index notation to TACO's loop nesting + workspace allocation, then lowers directly to Cranelift IR (no intermediate C code generation)
- **Workspace transformation**: Dense temporary for inner-loop accumulation, compressed output assembly
- **Kernel selection**: Dispatcher routes to format-specific CPU or GPU kernels (SpMV, SpMM, structured MMA.sp for Ampere 2:4)
- **Cost model integration**: Density-aware FLOP estimation for kernel selection

---

## vmap / Automatic Batching (M39)

**Codegen**: `vmap.rs`
**Semantic**: `vmap.rs`

```
@vmap(batch_dim=0)
fn per_sample_gradient(model, x, y):
    let loss = model.forward(x, y)
    return grad(loss)
```

Transforms a function written for single samples into one that operates on batches:
- **Batch tracking**: Marks tensors as batch-variant or batch-invariant
- **Dimension insertion**: Adds batch dimension to operations automatically
- **Binary op rules**: Determines output batch status from input batch statuses

---

## Multimodal Primitives (M48)

**Codegen**: `multimodal.rs`
**Runtime**: `multimodal.rs`

### Vision
- **Patch embedding**: Image → patch tokens (configurable patch size, embed dim)
- **Bilinear resize**: Image rescaling
- **Image normalization**: Channel-wise mean/std normalization
- **Cross-attention**: Vision-language cross-attention

### Audio
- **STFT**: Short-Time Fourier Transform
- **Mel spectrogram**: Mel filterbank construction and application
- **Hz/Mel conversion**: Frequency scale conversion utilities

---

## Unikernel Infrastructure (M54)

**File**: `nsl-codegen/src/unikernel.rs`

Configuration system for bare-metal deployment:

- **Target platforms**: KVM, Firecracker
- **Boot protocols**: Multiboot2, Linux boot
- **Weight sources**: Embedded in binary, disk, network fetch
- **Memory layout**: 60% model pool / 30% KV-cache / 10% heap
- **Linker script generation**: Custom memory regions for bare-metal
- **GPU init strategies**: Configurable for virtualized environments

Purpose: Deploy inference as a minimal virtual machine with no OS overhead — microsecond cold starts, no attack surface.

---

## Interoperability

### DLPack (M62)
**File**: `dlpack.rs`

Zero-copy tensor exchange with PyTorch, JAX, and other frameworks:
- DLPack v0.8 C ABI implementation
- Device type codes, dtype conversions
- `DLManagedTensor` lifecycle management

### C API (M62)
**File**: `c_api.rs`

Model lifecycle management for embedding NSL in other applications:
- `nsl_model_create()` / `nsl_model_destroy()`
- Error handling via thread-local storage
- DLPack tensor I/O

### SafeTensors I/O
**File**: `safetensors_io.rs`

Read/write HuggingFace SafeTensors format for weight loading.

### ONNX Export
**File**: `onnx.rs`

Export NSL models to ONNX format for deployment in other runtimes.

---

## Tensor Debugging (M45)

**Runtime**: `tensor_trace.rs`, `trace_diff.rs`

### Binary Trace Format
- 124-byte fixed-size `TraceEntry` per operation
- Fields: op_id, op_type, flags, timestamp_ns, 3× TensorStats (shape, dtype, device, min/max/mean/std)
- Header: magic "NSLT" + version
- Thread-local `TraceRecorder`

### Trace Diff
Differential testing between traces (e.g., fused vs. unfused execution) to verify optimization correctness.

```bash
nsl run --trace model.nsl    # generates trace file
# Trace can be loaded in Chrome's chrome://tracing
```

---

## NSL's Moat Features (Impossible in Python)

These features require static analysis of the full computation graph — something fundamentally incompatible with Python's dynamic dispatch:

| Feature | Why Python Can't | NSL Approach |
|---------|-----------------|--------------|
| Memory planning (M36) | Dynamic allocation → no bounds | Static liveness analysis |
| Roofline cost model (M37) | Unknown graph → no prediction | AOT graph → per-op estimation |
| Linear types (M38) | No ownership system | Rust-inspired move semantics |
| vmap (M39) | Dynamic dispatch → can't transform | Static graph → batch dimension insertion |
| Source-to-source AD (M40) | Dynamic tape → can't rewrite | Static graph → Wengert extraction |
| Compiled FSM (M44) | Interpreted regex per token | NFA→DFA→minimized→compiled |
| Determinism proofs (M46) | No effect tracking | Effect system + kernel variants |
| Shape algebra (M49) | No symbolic dimensions | Constraint solver + named dims |
| Effect system (M51) | No side-effect tracking | Bitset effects + call graph propagation |
| Weight-aware compilation (M52) | Weights loaded at runtime | Weights loaded at compile time |
| WCET proofs (M53) | Dynamic allocation + GC | Zero alloc + static instruction counts |
| ZK circuits (M55) | Dynamic graph → no circuits | Static DAG → PLONKish constraints |
