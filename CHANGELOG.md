# Changelog

All notable changes to NeuralScript will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.9.1] - 2026-03-26

### M41b: NVLink/RDMA/TCP KV Transfer Backends
- **TcpBackend**: TCP socket-based KV transfer for multi-node disaggregated inference (per-rank listener, retry logic, Nagle disabled)
- **NvlinkBackend**: CUDA IPC GPU-direct transfer for same-node multi-GPU (cuIpcGetMemHandle/cuIpcOpenMemHandle, falls back to staged CPU transfer)
- **RdmaBackend**: RDMA verbs-based zero-copy transport for HPC clusters (ibverbs memory registration, InfiniBand/RoCE hardware probe, TCP fallback)
- **Auto-detection**: `auto_select_backend()` probes NVLink > RDMA > TCP > SharedMem based on available hardware
- **Serve block wiring**: `kv_transfer` config string flows through codegen, workers emit `nsl_kv_transfer_init`/`destroy`

### M35b: GPTQ Full OBQ Algorithm
- **Optimal Brain Quantizer**: Column-wise quantization with Hessian-based error compensation (replaces RTN stub)
- **Hessian computation**: `HessianAccumulator` for X^T X calibration data accumulation
- **Cholesky factorization**: Damped Hessian inverse via Cholesky decomposition for numerical stability
- **Act-order**: Columns quantized in descending Hessian diagonal order for better quality
- **Blocked updates**: Lazy batch error propagation for memory efficiency on large matrices
- **Calibration FFI**: `nsl_gptq_hessian_init`, `nsl_gptq_hessian_add_batch`, `nsl_gptq_hessian_finalize`

### M54b: Bare-Metal Unikernel Boot Stub, Runtime & GPU Init
- **x86_64 boot stub generator**: Multiboot2 header, GDT (64-bit code/data segments), PML4/PDPT page tables (identity-map 4GB), SSE/AVX enable, long mode transition
- **Unikernel runtime**: Bump allocator (lock-free atomic), serial console (COM1 115200 8N1), boot config JSON parser
- **GPU init framework**: PCI bus scan (CF8h/CFCh), NVIDIA device discovery, VFIO passthrough path (cuInit), direct register path (BAR0 MMIO)
- **ELF image builder**: Combines boot stub + compiled code + weights + linker script into single binary

### Documentation
- Updated README.md with new CLI commands (unikernel, ZK), test count (1,558)
- Updated implementation status: 34 production milestones (was 30), 131,800 LOC across 282 files
- Updated CHANGELOG and SPECIFICATION

## [0.8.0] - 2026-03-18

### Consolidation & Code Quality
- **CLI flag wiring**: all CompileOptions (--no-autotune, --deterministic, --disable-fusion, --tape-ad, --trace-ops, --nan-analysis, --target) now flow from CLI to compiler
- **Refactored hotspot files**: tensor.rs (5K→6 files), expr.rs (3.5K→6), compiler.rs (2.8K→7), checker.rs (2.6K→8), autodiff.rs (1.8K→3)
- **Error handling**: replaced 14 panics in process spawning and FFI with graceful error codes
- **Parser**: generic trait bounds now parsed (not enforced yet); if-expression limitations documented
- **Deterministic scatter_add**: changed from silent null return to explicit abort with message
- **E2E precision**: float comparison tightened from 4 to 6 decimal places
- **Version**: workspace version aligned to release tags

### Phase 8–9 Infrastructure (analysis + FFI complete, codegen wiring in progress)
- **M45**: Tensor debugger — trace recording, NaN analysis, trace diffing, Chrome export
- **M46**: Reproducibility — determinism checker, kernel variant selection, RNG tracking
- **M48**: Multimodal — PatchEmbed, MelSpectrogram, cross_attention, modality classification
- **M49**: Shape algebra — symbolic dimension solver (equality, divisibility, range proofs)
- **M50**: Sparse tensors — NslSparseTensor, COO/CSR/CSC/BSR format dispatch

## [0.7.0] - 2026-03-18

### Phase 7: Distributed Training
- **M38b**: Linear types codegen — ownership decisions for tensor lifetime
- **M40b**: Source AD extraction — Wengert extraction from AST, backward context
- **M43**: Pipeline parallelism — 1F1B/GPipe scheduling, 3D rank mapping, ZeRO sharding

## [0.6.0] - 2026-03-18

### Phase 6: Deployment & Portability
- **M41**: Disaggregated inference — prefill/decode worker separation, KV transfer
- **M47**: Multi-backend KIR — Kernel IR, PTX backend, GpuTarget, GpuBackend trait
- **M39b**: vmap AST transform — VmapTransformer FnDef→FnDef rewriting
- Snapshot testing (insta) and differential testing infrastructure

## [0.5.0] - 2026-03-18

### Phase 5: Inference Optimization
- **M42**: KV-cache compression — INT8/INT4/FP8, sliding window, H2O eviction
- **M44**: Constrained decoding — compiled FSM, token-level DFA, logit masking

## [0.4.0] - 2026-03-18

### Phase 4 continued
- **M41**: Disaggregated inference (moved to Phase 6 delivery)

## [0.3.0] - 2026-03-17

### Phase 4: Scaling & Optimization (M32-M40)
- **M32**: Mixture of Experts — @moe, top-k gating, capacity routing, aux loss
- **M33**: Speculative Decoding — @speculative, tree attention, rejection sampling
- **M34**: Ring Attention — @context_parallel, cross-GPU sequence parallelism
- **M35**: FP8/AWQ/GPTQ quantization
- **M36**: Memory planning — compile-time liveness analysis, slab allocation
- **M37**: Roofline cost model — per-op FLOP/byte analysis
- **M38a**: Linear types semantics — ownership checker, @shared
- **M39a**: vmap analysis — batch tracking, shape rewriting, matmul classification
- **M40a**: Source AD analysis — Wengert list, adjoint rules, dead gradient elimination

## [0.2.0] - 2026-03-15

### Production Inference & Optimization (M23-M31)

#### M23: Custom Datatypes (BYOD)
- `datatype` block with `@pack`/`@unpack` methods for user-defined numeric formats
- Custom dtype registration with element-wise pack/unpack dispatch
- NslTensor.dtype expanded from u8 to u16 for custom dtype IDs

#### M24: Standalone Export
- `nsl build --standalone` produces zero-dependency native executables
- Embedded weights (bundled in binary) and sidecar weights (.nslweights file)
- WeightProvider abstraction with embedded and mmap backends
- Build-time safetensors reading for weight bundling

#### M25: PagedAttention & Memory Profiling
- Paged KV-cache with BlockAllocator, PageTable, and KvCacheManager
- `@paged_kv` decorator for automatic KV-cache management
- Memory watermark profiler with `--profile-memory` flag
- Chrome tracing JSON output for memory analysis

#### M26: @autotune, Fusion & Kernel Profiling
- `@autotune` decorator with Cartesian product search and build-time caching
- `@fuse` decorator for elementwise fusion chain detection
- Fused PTX synthesis for elementwise op chains
- Kernel profiler with Chrome tracing JSON (`--profile-kernels`)

#### M27: FlashAttention-2
- FlashAttention-2 PTX template synthesis with 5 kernel variants
- `scaled_dot_product_attention` lowering with naive and flash paths
- RoPE cache write kernels and GQA replication
- `@flash_attention`, `@rope`, `@gqa` decorator validation
- Shared memory parameter support in kernel_launch

#### M28: Dynamic Shapes & Bounded Dimensions
- Symbolic dimension tracking with `SymbolicDimTracker`
- Bounded dimension syntax (`SeqLen < 4096`) with parse/semantic/codegen support
- Runtime dimension assertions (`nsl_tensor_assert_dim`, `assert_dim_bound`)
- Dimension unification for Bounded and Computed dimensions

#### M29: Continuous Batching & Serving
- `serve` block language frontend (lexer, AST, parser, semantic, codegen)
- `BatchScheduler` with chunked prefill and `RaggedBatchBuilder`
- `PreemptionManager` with swap/recompute policies
- `InferenceRequest` lifecycle management

#### M30: Tensor Parallelism
- `@shard` decorator for weight distribution across GPUs
- `CollectiveBackend` trait with simulated backend for testing
- SPMD process launcher with `--devices` flag
- Tensor parallel FFI: init, rank, collectives (all-reduce, all-gather, broadcast), destroy
- Weight sharding with `compute_shard_slice` and `copy_shard`

#### M31: Graph-Level Operator Fusion
- `FusionGraph` DAG with ANF node model and consumer counting
- Epilogue fusion: matmul+bias+activation chain detection and PTX synthesis
- Reduction fusion: softmax, layernorm, rmsnorm pattern matching and PTX synthesis
- `@fuse_graph` and `@no_fuse` decorator validation
- `--fusion-report` CLI flag for fusion event logging

### Bug Fixes
- Fix use-after-free in autodiff backward for SumReduce/MeanReduce global reductions
- Add `in_tape_region` guard to suppress tensor temporary cleanup during tape recording
- Fix macOS platform version for Cranelift objects
- Fix macOS linker flags and E2E baselines
- Make interop non-default feature to avoid OpenSSL link dependency
- Numerous clippy warning fixes across all modules

## [0.1.0] - 2026-03-12

### Language Features
- Indentation-based syntax with Python-familiar keywords
- Pipe operator (`|>`) for model op chaining
- `let`/`const` variable declarations with type inference
- `fn` functions with named/default parameters
- `model` keyword for neural network definitions
- `grad` keyword for tape-based automatic differentiation
- `train` block DSL with declarative data/optimizer/scheduler
- `quant` block for INT4/INT8 weight quantization
- `kernel` keyword for custom GPU kernels (PTX codegen)
- Compile-time tensor shape checking with named dimensions
- `@no_grad`, `@checkpoint`, `@backward`, `@test` decorators
- Import system with multi-file compilation

### Standard Library
- **nsl.nn**: Linear, Embedding, Conv2d, MaxPool2d, LayerNorm, RMSNorm, Dropout, Attention, TransformerBlock
- **nsl.nn.activations**: relu, gelu, silu, sigmoid, tanh, softmax, elu
- **nsl.nn.losses**: mse_loss, l1_loss, cross_entropy, bce_loss
- **nsl.optim**: SGD, Adam, AdamW, Lion, Muon, SOAP
- **nsl.optim.schedulers**: constant_lr, step_lr, exponential_lr, linear_decay, cosine_anneal, warmup_cosine, one_cycle
- **nsl.tokenize**: byte_tokenizer, BPE encode/decode
- **nsl.data**: JSONL/CSV/mmap DataLoader with batching, shuffling, sequence packing
- **nsl.inference**: topk, multinomial, argmax, autoregressive generation
- **nsl.quant**: quantize/dequantize (INT4/INT8)
- **nsl.compat**: safetensors load/save, HuggingFace model loading, ONNX export

### Tooling
- `nsl run` -- compile and execute NSL programs
- `nsl build` -- compile to native executable
- `nsl check` -- type checking and semantic analysis
- `nsl test` -- run `@test` annotated functions
- `nsl export` -- ONNX model export
- `nsl fmt` -- code formatter
- `nsl init` -- project scaffolding

### GPU Support
- CUDA backend with 15 PTX kernels (elementwise ops + matmul)
- `kernel` keyword for custom GPU ops
- Device transfer (`.to(cuda)`, `.to(cpu)`)
- Unified memory via cuMemAllocManaged

### Interop
- Safetensors read/write
- HuggingFace Hub model loading (single + sharded)
- ONNX export

### Known Limitations
- No package manager or dependency resolution
- No PyTorch FFI (`to_torch`/`from_torch`)
- No distributed multi-GPU training (DDP)
- No REPL
- CUDA required for GPU features (no ROCm/Metal)
- Windows requires Visual Studio Build Tools for linking
