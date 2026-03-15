# Changelog

All notable changes to NeuralScript will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

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
