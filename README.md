# NeuralScript (NSL)

A statically-typed, compiled programming language designed as a first-class replacement for Python + PyTorch in AI/ML workloads.

NSL compiles to native code via Cranelift with zero Python or C++ dependencies. The entire stack is Rust (compiler + runtime) and NSL (standard library).

## Why NSL?

- **Python-familiar syntax** with indentation-based blocks, `let`/`const`, `fn`, `model`
- **Compile-time tensor shape checking** — catch dimension mismatches before running
- **Native autodiff** — `grad` is a keyword, not a library call
- **Declarative training** — `train` blocks replace boilerplate training loops
- **GPU/CUDA native** — `kernel` keyword for custom GPU ops, `.to(cuda)` for device transfer
- **No GIL** — just `nsl run model.nsl`

## Quick Example

```python
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
        let loss = mse_loss(pred, y)
    callbacks:
        on_step(step, loss):
            print(loss)
```

Output:
```
4.0
3.6864
3.39738624
3.131031158784
2.8855583159353344
```

### GPU Kernel Example

```python
kernel vec_add(a, b, c):
    let i = thread_id()
    c[i] = a[i] + b[i]

let a = full([1024], 1.0).to(cuda)
let b = full([1024], 2.0).to(cuda)
let c = zeros([1024]).to(cuda)
vec_add(a, b, c, grid=4, block=256)
let result = c.to(cpu)  # each element = 3.0
```

## Features

### Core Language
- Indentation-based syntax (no braces)
- `let` (mutable), `const` (immutable) bindings
- `int`, `float`, `bool`, `str`, `list`, `dict` types
- `fn` functions with type annotations
- `for`/`while` loops, `if`/`elif`/`else`
- Pattern matching, tuple destructuring
- Module system with `from ... import ...`

### Tensors
- `Tensor` type with shape-aware operations
- Creation: `zeros`, `ones`, `rand`, `full`, `arange`
- Arithmetic: `+`, `-`, `*`, `/`, `@` (matmul)
- Reductions: `sum`, `mean` (global or with `dim`/`keepdim`)
- Element-wise: `exp`, `log`, `sqrt`, `abs`, `sign`, `clamp`
- Shape ops: `reshape`, `transpose`, `clone`

### Models
- `model` keyword for defining neural network architectures
- Named parameters as typed fields
- Methods with `self` access
- Constructor with default values

### Automatic Differentiation
- `grad(params):` block records a computation tape
- Tape-based reverse-mode autodiff
- `@no_grad` decorator to exclude functions from tape
- Supports: add, sub, mul, div, matmul, exp, log, sqrt, abs, clamp, sum, mean, reduce_max, gather
- Broadcasting: NumPy-style broadcast for elementwise ops

### Neural Network Layers
- **Activation functions**: relu, gelu, silu, sigmoid, tanh, softmax
- **Layers**: Linear, Embedding, MLP, Attention, Conv2d, MaxPool2d
- **Normalization**: LayerNorm, RMSNorm
- **Regularization**: Dropout (training-mode aware)
- **Weight init**: randn for Kaiming-style initialization

### Training DSL
- `train(model=m, epochs=N):` declarative training blocks
- **6 optimizers**: SGD, Adam, AdamW, Lion, Muon, SOAP
- **7 schedulers**: constant_lr, step_lr, exponential_lr, linear_decay, cosine_anneal, warmup_cosine, one_cycle
- **4 loss functions**: mse_loss, l1_loss, cross_entropy, bce_loss
- Implicit gradient computation (no manual `grad` blocks needed)
- Callbacks: `on_step(step, loss)`, `on_epoch(epoch, loss)`
- `.nslm` checkpoint format for model serialization

### Tokenization
- Byte-level tokenizer (no training needed)
- BPE tokenizer via HuggingFace `tokenizers` crate
- Encode text → token ID tensors, decode back to text
- Batch encoding with padding and attention masks

### GPU/CUDA
- `kernel` keyword for user-defined GPU kernels (compiled to PTX)
- `.to(cuda)` / `.to(cpu)` for transparent device transfer
- GPU intrinsics: `thread_id()`, `block_id()`, `block_dim()`, `sync_threads()`
- Kernel launch with explicit grid/block: `my_kernel(a, b, c, grid=4, block=256)`
- 15 built-in GPU PTX kernels for tensor ops (add, mul, matmul, activations, etc.)
- CUDA Unified Memory for zero-copy host/device access
- Automatic f64↔f32 dtype conversion on device transfer
- Optional: build with `--features cuda` to enable GPU support

### Quantization
- `quant static` block for declarative weight quantization
- `QuantizedTensor` type with packed INT4 (2 values/byte) and INT8 storage
- Per-tensor, per-channel, and per-group granularity
- Asymmetric affine quantization (weight-only RTN)
- Mixed-precision matmul (`nsl_qtensor_matmul_mixed`)
- Model monomorphization: compiler synthesizes quantized model type
- Glob-based `exclude` patterns to keep specific layers in full precision
- **FP8 compute** — `@fp8_compute` decorator, E4M3/E5M2 scale management, FP8 Tensor Core matmul (sm_90 MMA + Hopper wgmma), running-max EMA calibration
- **AWQ 4-bit** — `quant { dtype: awq4 }` with in-register dequantize-in-GEMM
- **GPTQ 4-bit/8-bit** — `quant { dtype: gptq4 }` with Hessian-based optimal quantization

### Production Inference *(shipped — v0.2.0)*
- **Custom datatypes** — `datatype` block for user-defined numeric formats with PTX escape hatch
- **Standalone export** — `nsl build --standalone` for zero-dependency inference binaries
- **PagedAttention** — paged KV-cache with block allocation and CoW branching
- **FlashAttention-2** — tiled attention with online softmax, MMA tensor cores (sm_80 `mma.sync` + sm_90 `wgmma.mma_async`), RoPE/GQA fusion, paged KV integration, tree-structured causal mask, logsumexp backward storage
- **Dynamic shapes** — `Dim::Bounded` symbolic dimensions with stride-based codegen
- **Continuous batching** — `serve` block DSL with chunked prefill and preemption
- **Tensor parallelism** — `@shard` decorator with NCCL-backed all-reduce
- **Graph-level fusion** — epilogue fusion (bias+activation in matmul epilogue), reduction fusion (Welford merge for layernorm/softmax), `@fuse_graph` decorator, cost-guided profitability filtering
- **`@autotune`** — build-time kernel parameter tuning with real GPU benchmarking and persistent cache

### Scaling & Optimization *(shipped — v0.3.0)*
- **Mixture of Experts** — `@moe` decorator with top-k gating, capacity-based routing, aux loss
- **Speculative decoding** — `@speculative` with tree attention, rejection sampling, Medusa heads
- **Ring attention** — `@context_parallel` for cross-GPU sequence parallelism
- **Memory planning** — compile-time tensor liveness analysis, interference graph, BFD slab allocation with 256-byte alignment
- **Roofline cost model** — multi-level memory hierarchy (L1/L2/HBM bandwidth), occupancy estimation, per-op FLOP/byte analysis with GPU database (A100, H100, RTX-4090, etc.)
- **Cost-guided fusion** — profitability analysis (arithmetic intensity improvement threshold), register pressure estimation, prevents counterproductive fusions
- **Linear types semantics** — ownership checker for use-after-move detection, immutable borrow (`&T`) semantics, `@shared` escape hatch
- **vmap analysis** — `@vmap` batch tracking, shape rewriting, matmul rewrite classification
- **Source AD** — Wengert extraction, reverse-mode adjoint rules, if/else branch support with condition saving, dead gradient elimination

### Infrastructure *(v0.4.0–v0.8.0 — analysis + FFI layers + codegen wiring)*

- **Disaggregated inference** — router, KV transfer, prefill/decode worker FFI
- **KV-cache compression** — INT8/INT4/FP8 quantization, sliding window, H2O eviction
- **Constrained decoding** — compiled FSM (Thompson/subset/Hopcroft DFA), token alignment, logit masking, per-request grammar state
- **Multi-backend** — Kernel IR (40+ ops), PTX + AMDGPU + Metal (MSL) + WGSL backends
- **Pipeline parallelism** — 1F1B/GPipe scheduling, 3D rank mapping, ZeRO sharding, @pipeline codegen
- **Tensor debugger** — `nsl debug` CLI: trace reader, NaN finder, diff, Chrome export
- **Reproducibility** — determinism checker, CPU deterministic scatter_add, kernel variant selection
- **Multimodal** — patch embed, bilinear resize, image normalize, cross-attention, STFT, mel spectrogram, resampling
- **Sparse tensors** — COO/CSR construction, from_dense, to_dense, SpMM
- **Shape algebra** — symbolic dimension solver (equality, divisibility, range proofs)
- **Linear types codegen** — ownership decision tree (free-at-consumption, tape-holds-reference)
- **Source AD extraction** — Wengert extraction from AST, backward context
- **vmap AST transform** — VmapTransformer FnDef→FnDef rewriting
- **Effect system** — effect inference, @pure/@deterministic/@checkpoint validation
- **Snapshot + differential testing** — insta PTX snapshots, --disable-fusion oracle

### Weight Intelligence & Interop *(v0.9.0)*

- **Weight-aware compilation** — `nsl build --weights model.safetensors`: loads weights at compile time, sparsity analysis per matrix, constant folding through matmul/add/relu, dead weight elimination, SHA-256 integrity hash
- **DLPack zero-copy bridge** — PyTorch/JAX tensor exchange without copying data (DLPack v0.8)
- **C API** — `nsl build --shared-lib`: model lifecycle, forward pass, DLPack interop, error handling
- **Unikernel infrastructure** — `nsl build --unikernel`: memory layout computation, linker script generation, boot config

### Hopper GPU Optimization *(v0.9.0)*

- **wgmma.mma_async** — 128-thread warp group MMA on H100/H200 for both FlashAttention and FP8 matmul (~37% more tensor core utilization vs mma.sync)
- **FlashAttention backward** — logsumexp auxiliary storage, tiled backward kernel for dQ/dK/dV with attention probability recomputation
- **FP8 E5M2 backward** — mixed-precision training: E4M3 forward (higher precision) + E5M2 backward (wider dynamic range for gradients), GPU MMA kernels for both directions
- **Effect system Rng fix** — `@deterministic` now correctly allows explicit `Rng` parameter (controlled randomness)

### Test Framework
- `@test` decorator marks functions as test cases
- `assert_eq`, `assert_close` for value and tensor assertions
- `nsl test` CLI with process isolation (each test in separate process)
- Filter tests with `--filter`

## Installation

### From GitHub Releases (recommended)

Download the latest release for your platform from
[GitHub Releases](https://github.com/bwiemz/NSL/releases).

Extract the archive to a permanent location and add `bin/` to your PATH:

```bash
# Linux/macOS
tar xzf nsl-v0.9.0-<target>.tar.gz
export PATH="$PWD/nsl-v0.9.0-<target>/bin:$PATH"
```

> **Important:** Do not separate the `nsl` binary from the `lib/` directory.
> The compiler needs `lib/libnsl_runtime.a` for linking and `lib/stdlib/`
> for standard library imports.

### Prerequisites

- **Linux/macOS:** A C compiler (`gcc`, `clang`, or `cc`) -- usually pre-installed
- **Windows:** [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) (MSVC `link.exe`)
- **GPU (optional):** NVIDIA CUDA Toolkit for GPU features

### From Source

Requires Rust (stable).

```bash
cargo build -p nsl-cli
```

## Quick Start

```bash
nsl init myproject
cd myproject
nsl run main.nsl
```

## Usage

```bash
# Run an NSL program
nsl run examples/hello.nsl

# Compile to native executable
nsl build examples/hello.nsl

# Type-check without running
nsl check examples/hello.nsl

# Format code
nsl fmt examples/*.nsl

# Run tests
nsl test tests/m15_test.nsl

# Check version
nsl --version
```

## Project Structure

```
crates/
  nsl-lexer/       Tokenizer (indentation-aware)
  nsl-ast/         Abstract syntax tree definitions
  nsl-parser/      Recursive descent parser
  nsl-semantic/    Type checking, shape inference, name resolution
  nsl-codegen/     Cranelift IR generation and native compilation
  nsl-runtime/     Rust static library linked into every NSL binary
  nsl-cli/         Command-line interface (nsl run, nsl build)
  nsl-errors/      Shared error types

stdlib/nsl/        Standard library (written in NSL)
  math.nsl         Basic math utilities
  nn/layers.nsl    Linear, Embedding, MLP
  nn/norms.nsl     LayerNorm, RMSNorm
  nn/activations.nsl  Activation function wrappers
  nn/attention.nsl Dot-product attention
  nn/dropout.nsl   Dropout layer
  nn/losses.nsl    Loss functions (mse, l1, cross_entropy, bce)
  optim/           Optimizers (sgd, adam, adamw, lion, muon, soap)
  optim/schedulers.nsl  Learning rate schedulers
  tokenize/        Tokenizer wrappers
  quant/ops.nsl    Quantization wrappers

spec/              Language specification (13 chapters)
examples/          Example programs and integration tests
tests/expected/    Expected output for integration tests
docs/plans/        Design documents and implementation plans
```

## Testing

```bash
# Run unit tests
cargo test --workspace

# Run a specific integration test
cargo run -p nsl-cli -- run examples/m14_sgd_basic.nsl

# Run NSL test suite
cargo run -p nsl-cli -- test tests/m15_test.nsl

# Run end-to-end language model demo
cargo run -p nsl-cli -- run examples/m15_tiny_lm.nsl

# Run quantization example
cargo run -p nsl-cli -- run examples/m16_quantize.nsl

# Run GPU kernel test (requires CUDA)
cargo run -p nsl-cli --features cuda -- run tests/m17_kernel_test.nsl

# Run GPU compute test (requires CUDA)
cargo run -p nsl-cli --features cuda -- run tests/m17_gpu_training_test.nsl
```

### Compile-Time Analysis

```bash
nsl check --perf file.nsl          # Roofline performance analysis
nsl check --nan-analysis file.nsl  # NaN/Inf risk detection
nsl check --deterministic file.nsl # Determinism verification
nsl check --wcet file.nsl          # Worst-case execution time
nsl check --weight-analysis file.nsl --weights model.safetensors
```

## Roadmap

**Next priorities (unimplemented):**

- FBIP (Functional But In-Place) — zero-allocation inference via refcount-checked in-place mutation
- Effect polymorphism — effect variables on function types for correct higher-order effect propagation
- Shape algebra bounds — Fourier-Motzkin elimination for symbolic dimension proofs
- Memory planner rematerialization — trade compute for memory in training
- WCET two-tier — FPGA certified path (GPU WCET is statistically bounded only)
- ZK upgrade — lookup-native arithmetization (Jolt-style), folding accumulation, Mersenne-31 field
- Sparse merge lattices — TACO-style co-iteration for sparse tensor operations

**Future milestones (M56-M62):** Multi-agent shared memory, FPGA/neuromorphic backend, elastic fault tolerance, topology-aware routing, exabyte data streaming, cluster debugging, PyTorch FFI.

See `docs/plans/` and `docs/summaries/` for details.

## Known Limitations

- No REPL
- CUDA required for GPU features (ROCm/Metal/WebGPU KIR foundation built, backends in progress)
- Windows requires Visual Studio Build Tools for linking
- GPU WCET provides statistical bounds only — hard real-time certification requires FPGA target (not yet implemented)
- ZK circuits use v1 Halo2 scaffolding — upgrade to lookup-native/folding in progress
- Autodiff backward rules cover ~11 operations (need ~40+ for full coverage)
- Multi-backend (AMDGPU/Metal/WGSL) are codegen stubs — only PTX is production

## License

Apache 2.0
