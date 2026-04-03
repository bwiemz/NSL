# NeuralScript (NSL) — Technical Specification

Comprehensive feature reference, architecture details, and implementation status for NeuralScript v0.9.1.

For getting started, see [README.md](README.md).

---

## Architecture

NSL is a pure Rust+NSL stack with no C/C++/Python dependencies:

| Layer | Implementation | Role |
|-------|---------------|------|
| Lexer | Rust (`nsl-lexer`) | Indentation-aware tokenization |
| Parser | Rust (`nsl-parser`) | Recursive descent → AST |
| Semantic Analyzer | Rust (`nsl-semantic`) | Type checking, shape inference, ownership |
| Code Generator | Rust (`nsl-codegen`) | AST → Cranelift IR → native code |
| Runtime | Rust (`nsl-runtime`) | Tensor ops, autodiff, memory, GPU dispatch |
| Standard Library | NSL (`stdlib/nsl/`) | Neural network layers, optimizers, losses |

The compiler produces native executables via Cranelift JIT or AOT compilation. GPU code is emitted as PTX and launched via the CUDA driver API.

---

## Core Language

### Types and Syntax
- Indentation-based syntax (no braces), Python-familiar
- `let` (mutable), `const` (immutable) bindings
- Primitive types: `int`, `float`, `bool`, `str`
- Collections: `list`, `dict`, `tuple`
- `fn` functions with type annotations and return types
- `for`/`while` loops, `if`/`elif`/`else`, `match`/`case`
- Pattern matching with tuple destructuring
- Module system: `from nsl.nn.layers import Linear`
- Pipe operator: `x |> normalize |> relu`

### Tensor System
- `Tensor` type with compile-time shape checking
- Named dimensions: `Tensor<[batch="B", seq="S"], f32>`
- Creation: `zeros`, `ones`, `rand`, `randn`, `full`, `arange`, `linspace`
- Arithmetic: `+`, `-`, `*`, `/`, `@` (matmul), `**` (power)
- Reductions: `sum`, `mean`, `reduce_max` (global or per-dim with keepdim)
- Element-wise: `exp`, `log`, `sqrt`, `abs`, `sign`, `clamp`, `neg`
- Shape ops: `reshape`, `transpose`, `squeeze`, `unsqueeze`, `contiguous`, `expand`
- Comparison: element-wise `>`, `<`, `>=`, `<=`, `==`
- Device: `.to(cuda)`, `.to(cpu)` for transparent GPU transfer
- Data format: f64 (CPU default), f32 (GPU/training default), FP8/INT8/INT4 (quantized)

---

## Neural Network Features

### Model Definition
```python
model TransformerBlock(d_model: int, n_heads: int):
    attn_norm: RMSNorm = RMSNorm(d_model)
    attn: GroupedQueryAttention = GroupedQueryAttention(d_model, n_heads, 4, 0.1)
    ffn: SwiGLUFFN = SwiGLUFFN(d_model, d_model * 4, 0.1)

    fn forward(self, x: Tensor, training: bool) -> Tensor:
        let h = x + self.attn.forward(self.attn_norm.forward(x), training)
        return h + self.ffn.forward(h, training)
```

### Standard Library Layers
| Module | Contents |
|--------|----------|
| `nsl.nn.layers` | Linear, Embedding, MLP, Conv2d, MaxPool2d |
| `nsl.nn.norms` | LayerNorm, RMSNorm, BatchNorm |
| `nsl.nn.attention` | ScaledDotProductAttention, MultiHeadAttention |
| `nsl.nn.gqa` | GroupedQueryAttention (GQA with RoPE) |
| `nsl.nn.rope` | RotaryEmbedding (RoPE positional encoding) |
| `nsl.nn.activations` | relu, gelu, silu, sigmoid, tanh, softmax |
| `nsl.nn.losses` | mse_loss, l1_loss, cross_entropy, bce_loss |
| `nsl.nn.dropout` | Dropout (training-mode aware) |

### Training DSL
```python
train(model=m, epochs=100, grad_clip=1.0):
    data:
        let loader = DataLoader(dataset, batch_size=32, seq_len=1024)
    optimizer: AdamW(lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01)
    scheduler: warmup_cosine(warmup_steps=1000, min_lr=0.0001)
    step(batch):
        let pred = m.forward(batch.input_ids)
        let loss = cross_entropy(pred, batch.labels)
    callbacks:
        on_step(step, loss):
            print(loss)
        on_epoch(epoch, loss):
            model_save(m, "checkpoint.nslm")
```

**6 optimizers:** SGD, Adam, AdamW, Lion, Muon, SOAP
**7 schedulers:** constant_lr, step_lr, exponential_lr, linear_decay, cosine_anneal, warmup_cosine, one_cycle
**Gradient clipping:** configurable via `grad_clip=N` (default: no clipping)
**Gradient accumulation:** `grad_accumulation=N` for effective batch size scaling

### Automatic Differentiation
- **Tape-based reverse-mode AD** — default backend for `grad(...)` and `train(...)`
- **Source-to-source AD** — opt-in via `--source-ad` for `train(...)` and standalone `grad(...)`, with diagnostic fallback to tape AD when extraction or target resolution is unsupported
- **Supported ops:** matmul, add, sub, mul, div, exp, log, sqrt, abs, clamp, relu, gelu, silu, sigmoid, tanh, softmax, layernorm, rmsnorm, dropout, conv2d, maxpool2d, embedding_lookup, gather, scatter, attention, RoPE
- **`@no_grad`** decorator to exclude functions from tape
- **`@checkpoint`** decorator for activation checkpointing

---

## GPU/CUDA

### Kernel Definition
```python
kernel matmul_naive(A, B, C, M, K, N):
    let row = block_id() * block_dim() + thread_id()
    if row < M:
        for col in range(N):
            let sum = 0.0
            for k in range(K):
                sum = sum + A[row * K + k] * B[k * N + col]
            C[row * N + col] = sum
```

### GPU Features
- `kernel` keyword compiles to PTX via Cranelift
- 15 built-in PTX kernels (add, sub, mul, div, neg, relu, exp, log, sqrt, abs, sign, sigmoid, tanh, scalar ops, matmul)
- CUDA Unified Memory for zero-copy host/device access
- Automatic f64/f32 dtype conversion on device transfer
- GPU intrinsics: `thread_id()`, `block_id()`, `block_dim()`, `sync_threads()`
- `@autotune` decorator for build-time kernel parameter tuning

---

## Quantization

| Format | Precision | Method |
|--------|-----------|--------|
| INT8 | 8-bit integer | Per-tensor/per-channel affine |
| INT4 | 4-bit integer | Packed (2 values/byte) |
| FP8 E4M3 | 8-bit float (forward) | Per-tensor scale with EMA calibration |
| FP8 E5M2 | 8-bit float (backward) | Wider dynamic range for gradients |
| MXFP8 | Block FP8 | E8M0 scale per 32 elements (Blackwell) |
| AWQ 4-bit | 4-bit | Activation-aware weight quantization |
| GPTQ 4-bit | 4-bit | Hessian-based optimal quantization |

```python
quant static:
    dtype: int4
    granularity: per_group(128)
    exclude: "*.norm*"
```

---

## Production Inference (v0.2.0)

### PagedAttention (M25)
- Block-based KV-cache allocation (configurable block size)
- Copy-on-Write for beam search branching
- Block table mapping: logical → physical blocks
- 19 unit tests covering allocator, CoW, eviction

### FlashAttention (M27)
- `scaled_dot_product_attention()` intrinsic
- CPU naive path (correctness-verified against manual computation)
- GPU path: tiled attention with online softmax, MMA tensor cores
- Causal masking support
- Hopper wgmma path exists, but parts of the Hopper-specialized TMA/wgmma kernel plumbing are still placeholder-level
- Logsumexp save/backward support is implemented

### Continuous Batching (M29)
- `serve` block DSL with `@endpoint` decorator
- Chunked prefill with configurable chunk size
- Request scheduling with preemption (swap/recompute policies)
- Ragged batch support for variable-length sequences
- Grammar-constrained decoding (compiled FSM)

### Speculative Decoding (M33)
- `@speculative` decorator with configurable draft models
- EAGLE-2 dynamic confidence-scored draft tree scaffolding
- Medusa multi-head speculation scaffolding
- Tree attention masks for parallel verification
- Rejection sampling (greedy + stochastic)
- Disaggregated serve/decode wiring supports the basic speculative lookahead path; tree-specific method/tree-width controls are still not fully wired end to end

---

## Scaling & Optimization (v0.3.0)

### Mixture of Experts (M32)
- `@moe` decorator with top-k gating
- Capacity-based routing with auxiliary load-balancing loss
- Expert routing/dispatch, with expert-parallel all-to-all still stubbed

### Memory Planning (M36)
- Compile-time tensor liveness analysis
- Interference graph coloring
- Best-Fit Decreasing slab allocation (256-byte aligned)
- Rematerialization pass (trade compute for memory)

### Roofline Cost Model (M37)
- Multi-level memory hierarchy modeling (L1/L2/HBM bandwidth)
- GPU database: A100, H100, RTX-4090 specifications
- Per-op FLOP/byte analysis with bound classification
- 50 unit tests validating all formulas
- Fusion profitability estimation

### Graph-Level Fusion (M31)
- Status: analysis is implemented and selective rewrite paths exist, but end-to-end graph-pass integration remains uneven and some decorator/training-path follow-ups remain

### Ownership & Linear Types (M38)
- Use-after-move detection at compile time
- Immutable borrow (`&T`) syntax with auto-borrow
- `@shared` escape hatch for explicitly shared tensors
- FBIP (Functional But In-Place): refcount==1 check enables zero-allocation mutation
- Runtime retain/release for FBIP safety on named variables
- Autodiff tape transparency for borrowed tensors

### Source-to-Source AD (M40)
- Wengert list extraction from AST
- Reverse-mode adjoint generation
- Enabled for `train(...)` and standalone `grad(...)` under `--source-ad`
- If/else branch support with condition saving
- Unsupported extraction, unresolved grad targets, or lowering failures emit a diagnostic and fall back to tape AD instead of changing gradient semantics
- Dead gradient elimination

---

## Infrastructure (v0.4.0–v0.8.0)

### Disaggregated Inference (M41)
- Router with prefill/decode worker pools
- KV transfer protocol (KVXF format with dtype support)
- Structured worker configs are threaded into prefill/decode loops, including speculative-lookahead metadata used by disaggregated serve
- Decode limits are enforced over total sequence length (`prompt_len + generated_tokens`); `max_tokens: 0` completes immediately
- Policies: LeastLoaded, RoundRobin, MemoryAware

### Multi-Backend (M47)
- Kernel IR with 40+ ops
- PTX backend (production)
- AMDGPU, Metal (MSL), WGSL backends (code generation, untested on hardware)

### Pipeline Parallelism (M43)
- 1F1B and GPipe scheduling
- 3D rank mapping (data, tensor, pipeline)
- SharedMem mailbox backend with condvar signaling

### Tensor Debugger (M45)
- `nsl debug` CLI for trace analysis
- NaN/Inf finder, tensor diff, Chrome trace export

### Effect System (M51)
- 4 effect categories: IO, Random, Mutation, Communication
- Effect inference via call graph propagation
- `@pure`, `@deterministic` decorators
- Effect polymorphism for generic functions

### Shape Algebra (M49)
- Symbolic dimension solver
- Equality, divisibility, and range proofs
- Fourier-Motzkin bound reasoning for inequality chains

### Sparse Tensors (M50)
- COO, CSR, CSC, BSR formats with conversion
- SpMV, SpMM, sparse add/mul
- TACO-style merge lattices (intersection for mul, union for add)
- Workspace transformation for output assembly
- `@sparse(pattern="2:4")` decorator for Ampere structured sparsity

---

## Weight Intelligence & Interop (v0.9.0)

### Weight-Aware Compilation (M52)
- `nsl build --weights model.safetensors`
- Sparsity analysis per weight matrix
- Constant folding through matmul/add/relu
- Dead weight elimination
- SHA-256 integrity verification

### Legacy Interop (M62)
- DLPack v0.8 zero-copy tensor bridge (PyTorch/JAX)
- C API: `nsl build --shared-lib` for model lifecycle management
- Safetensors import/export

### Unikernel (M54)
- `nsl build --unikernel` for bare-metal deployment
- Memory layout computation and linker script generation
- Hypervisor targets: KVM, Firecracker

### ZK Inference (M55)
- `@zk_proof(mode="weight_private")` decorator
- 4 privacy modes: weight_private, input_private, full_private, architecture_attestation
- Circuit IR, witness generation, Halo2 + Plonky3 backends
- Lookup-native arithmetization (Jolt-style)
- Mersenne-31 field support

### WCET Analysis (M53)
- Real-time execution time bounds for safety-critical deployment
- Op-by-op worst-case classification
- DO-178C reporting format

---

## Compile-Time Analysis Tools

```bash
nsl check --perf file.nsl                               # Roofline analysis
nsl check --nan-analysis file.nsl                        # NaN/Inf risk detection
nsl check --deterministic file.nsl                       # Non-determinism detection
nsl check --wcet file.nsl                                # Worst-case execution time
nsl check --weight-analysis file.nsl --weights model.st  # Weight sparsity analysis
nsl run file.nsl --disable-fusion                        # Differential testing
nsl run file.nsl --trace-ops                             # Tensor operation tracing
nsl run file.nsl --deterministic                         # Deterministic mode
```

---

## Benchmark Results (March 2026)

### Operator Fusion (M31)

Elementwise chains are fused into a single loop pass — zero intermediate allocations.

| Chain | Tensor Shape | Fused Time | Unfused Time | Speedup | Memory Saved |
|-------|-------------|------------|--------------|---------|-------------|
| `sigmoid(relu(a + b))` | [1000, 1000] | 3.45 ms | 8.67 ms | **2.51x** | 50% (4 MB vs 8 MB) |
| `sigmoid(tanh(relu(a + b)))` | [256, 512] | 32.3 ms (20 iters) | 52.7 ms | **1.63x** | 50% (10 MB vs 20 MB) |
| FFN block (matmul+bias+relu) | [256, 128]→[256, 512] | 53.3 ms (20 iters) | 74.5 ms | **1.40x** | Same (matmul dominates) |

### DataLoader I/O Throughput (M19)

After removing redundant attention mask construction (134 MB/batch waste):

| Config | Before Fix | After Fix | Improvement |
|--------|-----------|-----------|-------------|
| batch=32, seq=1024 | 29 batches/sec, 941K tok/s | **9,335 batches/sec, 306M tok/s** | **322x** |
| batch=1, seq=1024 | ~1K batches/sec, 1M tok/s | **270K batches/sec, 277M tok/s** | **270x** |

### Roofline Cost Model — NSLCoder-50M (M37)

Per-op analysis targeting H100-SXM (FP32, batch=1, seq=1024):

| Operation | GFLOPs | AI (FLOP/byte) | Bound | Est. Time (us) |
|-----------|--------|-----------------|-------|----------------|
| Q/K/V projections | 0.27-0.54 | 73-102 | Compute | 1.1-1.6 |
| Attention QK^T | 1.07 | 28.4 | Compute | 11.3 |
| **Softmax** | 0.04 | **0.625** | **Memory** | **20.0** |
| FFN gate/up/down | 1.48 each | 137.4 | Compute | 3.2 each |
| **LM head** | **51.5** | 169.5 | Compute | **90.8** |

| Precision | Total GFLOPs | Overall AI | Compute Time (us) | Memory Time (us) | Bottleneck |
|-----------|-------------|------------|-------------------|------------------|------------|
| **FP32** | 117.5 | 63.9 | 118.8 | **549.0** | **Memory** |
| **FP8** | 117.5 | 255.5 | 59.4 | **137.3** | **Memory** |
| FP8 vs FP32 speedup | — | 4x higher AI | 2.0x | **4.0x** | — |

### Other Benchmarks

| Benchmark | Result | Notes |
|-----------|--------|-------|
| Training correctness (SGD) | loss: 4.0 → 2.89 (5 epochs) | Mathematically verified |
| FlashAttention correctness | Naive == SDPA output exactly | `scaled_dot_product_attention` intrinsic |
| Serving infrastructure | 32 tests pass | PagedKV, CoW, scheduler, preemption |
| Cost model formulas | 50/50 tests pass | All FLOP/bandwidth formulas validated |
| Test suite | **1,420 tests, 0 failures** | Across 7 crates |

---

## Roadmap

### Completed (M9-M55)
Milestone work from M9-M55 is present in the repo, with maturity ranging from production to functional and a few subsystems still relying on stubs or fallbacks. See [docs/summaries/](docs/summaries/) for the current per-milestone status.

### In Progress
- **NSL-Coder-50M** — 50M parameter code generation model (pretraining verified)

### Future (M56-M71)
| Phase | Milestones | Theme |
|-------|-----------|-------|
| v1.0 | M56-M62 | Multi-agent, FPGA, elastic fault tolerance, cluster debugging |
| v1.1 | M63-M64 | Compiled MCTS tree search, online DPO alignment |
| v1.2 | M65-M67 | Ternary 1.58-bit types, format-agnostic sparsity, neuromorphic |
| v1.3 | M68-M71 | Refinement types (Z3 SMT), phase-split inference, unikernels, universal ZKML |

See [docs/plans/](docs/plans/) for detailed designs.

---

## License

Apache 2.0
