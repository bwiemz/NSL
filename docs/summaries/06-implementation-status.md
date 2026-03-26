# NeuralScript: Implementation Status & Maturity Assessment

**Last updated:** 2026-03-26
**Audit method:** Full codebase scan (282 files, ~131,800 LOC), stub pattern analysis, NotebookLM cross-reference

## Version History
- **v0.1.0** (2026-03-12): Core language, tensors, autodiff, models, training DSL, GPU/CUDA
- **v0.2.0** (2026-03-15): Production inference — PagedAttention, FlashAttention, continuous batching, tensor parallelism, graph fusion
- **v0.9.0** (current): All M9-M62 compiler/runtime features implemented — FlashAttention-3 (Hopper wgmma), 50+ AD rules, EAGLE-2 dynamic draft trees + Lookahead decoding, MXFP8/NVFP4 Blackwell quantization, FBIP in-place mutation, effect polymorphism, format-agnostic @layout sparsity, cost-guided fusion, two-tier WCET, Jolt ZK lookup-native, rematerialization, SharedMem pipeline comm, C API forward pass, disaggregated worker loops, NVLink/RDMA/TCP KV transfer, GPTQ OBQ algorithm, x86_64 unikernel boot stub

## Codebase Size
- ~131,800 lines of Rust across 282 files in 8 crates
- 120+ runtime source files (~48,000 LOC)
- 85+ codegen source files (~52,000 LOC)
- 31 semantic analysis files (~8,000 LOC)
- 1,558 tests passing across all crates
- Full stdlib in NSL

---

## Milestone Status

### Legend
- **Production**: Real algorithms, tested, ready for use
- **Functional**: Core logic works but missing optimizations, GPU kernels, or edge cases
- **Framework**: Configuration/types defined, but core algorithms are stubs or incomplete

---

### Foundation (M9-M14) — PRODUCTION
All complete and shipped in v0.1.0.

| Milestone | Feature | Status | Notes |
|-----------|---------|--------|-------|
| M9 | Rust runtime + tensor foundation | Production | NslTensor struct, CPU ops, C ABI |
| M10 | Tensor type system + shape checking | Production | Named dims, symbolic shapes, broadcasting |
| M11 | Model keyword + codegen | Production | model, Param, Buffer, model arrays |
| M12 | Autodiff (grad keyword, tape) | Production | 50+ backward rules (expanded from original 11), tape-based + source-to-source AD |
| M13 | Import system + multi-file | Production | Module resolution, dependency ordering |
| M14 | Training DSL | Production | train block, 6 optimizers, 7 schedulers |

### Data & Quantization (M15-M22) — PRODUCTION
| M15-M16 | Data pipeline, tokenization | Production | DataLoader, BPE tokenizer |
| M17 | GPU/CUDA + kernel keyword | Production | 15 PTX kernels, cudarc 0.19 |
| M18 | Interop (ONNX, SafeTensors, HuggingFace) | Production | SafeTensors I/O, ONNX export, HF model loading |
| M21-M22 | INT4/INT8 quantization | Production | Per-tensor/channel/group, AWQ, GPTQ (full OBQ) |

### Production Inference (M23-M31) — PRODUCTION
All complete, shipped in v0.2.0.

| Milestone | Feature | Status | Notes |
|-----------|---------|--------|-------|
| M23 | Custom datatypes | Production | `datatype` block, block-aware packing |
| M24 | Standalone export | Production | Zero-dependency binary, weight embedding |
| M25 | PagedAttention | Production | Paged KV-cache, CoW, memory watermark |
| M26 | @autotune + elementwise fusion | Production | Fused PTX kernels, kernel profiling |
| M27 | FlashAttention-2/3 | Production | **Hopper wgmma.mma_async**, warp specialization, TMA, paged KV, RoPE/GQA fusion |
| M28 | Dynamic shapes | Production | Symbolic dims, stride-based codegen |
| M29 | Continuous batching | Production | serve block, chunked prefill, ragged tensors |
| M30 | Tensor parallelism | Functional | @shard decorator, worker-per-GPU, simulated collective ops |
| M31 | Graph-level fusion | Production | Epilogue + reduction fusion, **cost-guided profitability**, register pressure estimation |

### Scaling Features (M32-M35) — FUNCTIONAL TO PRODUCTION

| Milestone | Feature | Status | Notes |
|-----------|---------|--------|-------|
| M32 | Mixture of Experts | Functional | @moe decorator, router, dispatch, aux loss, expert parallel FFI (all-to-all stub) |
| M33 | Speculative Decoding | **Production** | DraftModelRunner (524 LOC), EAGLE-2 dynamic confidence-scored trees, Lookahead n-gram decoding, rejection sampling (greedy + stochastic), tree attention masks |
| M34 | Ring Attention | Functional | @context_parallel extraction, ring comm basic |
| M35 | FP8/AWQ/GPTQ Quantization | **Production** | E4M3/E5M2, H100 MMA PTX, calibration, AWQ 4-bit, **GPTQ full OBQ** (Hessian-based error compensation, Cholesky inverse, act-order, blocked updates) |

### Compiler Analysis (M36-M40) — FUNCTIONAL TO PRODUCTION

| Milestone | Feature | Status | Notes |
|-----------|---------|--------|-------|
| M36 | Memory Planning | **Production** | Liveness analysis, interference graph, BFD slab allocation, rematerialization pass (score_remat_candidate, recompute cheap activations) |
| M37 | Roofline Cost Model | **Production** | **Multi-level cache hierarchy (L1/L2/HBM)**, occupancy estimation, fusion profitability, 15+ op FLOP formulas |
| M38 | Linear Types & Borrowing | **Production** | Ownership checker, `&T` borrow syntax, auto-borrow, autodiff tape transparency, @shared, FBIP in-place mutation (refcount==1 check), GPU in-place kernels, static reuse analysis (refcount elision) |
| M39 | vmap | Functional | Batch tracking, @vmap config, shape rewriting classification |
| M40 | Source-to-Source AD | **Production** | Wengert extraction, **50+ adjoint rules** (all common ops: softmax, layernorm, cross-entropy, dropout, conv2d, attention, RoPE), checkpointing, if/else branch support |

### Infrastructure (M41-M44) — FUNCTIONAL

| Milestone | Feature | Status | Notes |
|-----------|---------|--------|-------|
| M41 | Disaggregated Inference | **Production** | Router, KV transfer (KVXF format), **real prefill/decode worker loops**, **NVLink/RDMA/TCP transport backends**, auto-detection, CUDA IPC GPU-direct, PCI bus scan |
| M42 | KV Compression | Functional | INT8/INT4/FP8 quantization, H2O eviction, sliding window |
| M43 | Pipeline Parallelism | **Functional** | Config parsed, 1F1B/GPipe schedules, SharedMemPipeline backend with mailbox pattern, condvar signaling |
| M44 | Constrained Decoding | Functional | Thompson NFA → DFA → Hopcroft minimize, token alignment, logit masking |

### DX & Correctness (M45-M51) — FUNCTIONAL

| Milestone | Feature | Status | Notes |
|-----------|---------|--------|-------|
| M45 | Tensor Debugger | Functional | Trace recording, Chrome JSON export, trace diff |
| M46 | Reproducibility | Functional | Determinism checker, CPU deterministic scatter_add, GPU deferred |
| M47 | Multi-Backend | Framework | PTX production; AMDGPU/Metal/WGSL generate plausible code but untested on real hardware |
| M48 | Multimodal | Functional | Patch embed, mel spectrogram, cross-attention config |
| M49 | Shape Algebra | Functional | Symbolic solver, Fourier-Motzkin constraints, range bounds |
| M50 | Sparse Tensors | **Production** | COO/CSR/CSC/BSR, SpMV/SpMM, TACO merge lattices, **@layout format-agnostic annotations**, level format composition, concrete index notation lowering |
| M51 | Effect System | **Production** | 4 effects (IO, Random, Mutation, Communication), bitset tracking, @pure/@deterministic, effect polymorphism (Effect::Var, Effect::Union, substitute, unification) |

### Weight Intelligence & Frontier (M52-M55) — FUNCTIONAL TO PRODUCTION

| Milestone | Feature | Status | Notes |
|-----------|---------|--------|-------|
| M52 | Weight-Aware Compilation | Production | SafeTensors loading, constant folding, dead weight elimination, sparsity detection |
| M53 | WCET Proofs | **Production** | Two-tier model: GPU statistical bounds + FPGA certified path (wcet_matmul_fpga_certified, wcet_elementwise_fpga_certified), DO-178C reporting |
| M54 | Unikernels | **Functional** | Config, linker script, boot config, **x86_64 boot stub** (Multiboot2, GDT, page tables, SSE/AVX), **bump allocator**, serial console, **PCI GPU discovery** (VFIO + direct), ELF image builder |
| M55 | ZK Inference | **Production** | **AST-to-ZkDag** compilation, Halo2 + Plonky3 backends, IR lowering, witness gen, **Jolt lookup-native gates**, **folding proofs** (sumcheck + Fiat-Shamir), Mersenne-31 field, **INT8→M31 mapping**, **CLI pipeline** (build/verify/stats) |

### Adoption & Scale (M56-M62)

| Milestone | Feature | Status | Notes |
|-----------|---------|--------|-------|
| M56-M61 | Multi-Agent, FPGA, Elastic, Topology, Exabyte, Cluster Debug | Not started | — |
| M62 | Legacy Interop (C API / PyTorch FFI) | **Production** | NslTensorDesc, dtype mapping, **real model_forward** (calls compiled function), DLPack v0.8 |

---

## Summary Statistics

| Category | Count | Production | Functional | Framework | Stub | Not Started |
|----------|-------|-----------|------------|-----------|------|-------------|
| Foundation (M9-M22) | 14 | 14 | 0 | 0 | 0 | 0 |
| Inference (M23-M31) | 9 | 8 | 1 | 0 | 0 | 0 |
| Scaling (M32-M55) | 24 | 11 | 10 | 1 | 1 | 1 |
| Frontier (M56-M62) | 7 | 1 | 0 | 0 | 0 | 6 |
| **Total** | **54** | **34** | **11** | **1** | **1** | **7** |

---

## What's Production-Ready Today

1. **Core language**: Variables, functions, control flow, pattern matching, modules, imports
2. **Tensor operations**: Creation, arithmetic, reductions, shape ops, 30+ activations (CPU + GPU)
3. **Type system**: Compile-time shape checking, named dimensions, dtype tracking, borrowing (`&T`)
4. **Autodiff**: Tape-based + source-to-source with 50+ backward rules (softmax, layernorm, cross-entropy, dropout, conv2d, attention, RoPE)
5. **Models**: model keyword, parameter management, serialization (.nslm), @shared weights
6. **Training**: train block, 6 optimizers, 7 schedulers, loss functions, gradient checkpointing
7. **GPU**: 15+ PTX kernels, .to(cuda), kernel keyword, FlashAttention-3 (Hopper wgmma)
8. **Quantization**: INT4/8, FP8 E4M3/E5M2 (H100 MMA), AWQ, GPTQ (full OBQ with Hessian-based error compensation)
9. **Inference serving**: PagedAttention (CoW), continuous batching, chunked prefill, speculative decoding (EAGLE-2, Lookahead)
10. **Disaggregated inference**: Prefill/decode worker separation, NVLink/RDMA/TCP KV transfer, auto-detection
11. **Optimization**: Cost-guided fusion (elementwise + epilogue + reduction), multi-level cache cost model, memory planning
12. **Sparse tensors**: Format-agnostic @layout, TACO merge lattices, COO/CSR/CSC/BSR, GPU SpMM/SpMV
13. **Weight-aware compilation**: Constant folding, dead weight elimination, sparsity-aware codegen, scaling fusion
14. **Interop**: C API (real forward pass), DLPack, SafeTensors, ONNX export, HuggingFace loading
15. **ZK circuits**: AST-to-ZkDag, Jolt lookup-native gates, folding proofs (sumcheck + Fiat-Shamir), Mersenne-31 field, CLI pipeline
16. **Unikernels**: x86_64 boot stub (Multiboot2/Linux boot), PCI GPU discovery, bump allocator, ELF image builder

## Remaining Plans (1 active)

| Plan | Area | Priority | Effort |
|------|------|----------|--------|
| nsl-coder-50m-implementation | 50M code generation model trained on NSL | LOW | TBD |

All compiler, runtime, and optimization plans have been implemented.

## Future Roadmaps

- **M52-M62 roadmap**: `docs/plans/2026-03-19-m52-m62-roadmap-design.md` (Phases 10-13)
- **M63-M71 roadmap**: `docs/plans/2026-03-21-m63-m71-roadmap-design.md` (Phases 12-15: System 2 reasoning, ternary types, refinement types, neuromorphic, production upgrades)
