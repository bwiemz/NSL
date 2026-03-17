# NeuralScript Development Roadmap: M32-M51

**Date:** 2026-03-15
**Status:** Draft (revised ordering)
**Prerequisite:** v0.2.0 (M23-M31 complete)
**Architecture:** Rust compiler + Rust runtime + NSL stdlib + Cranelift AOT + PTX GPU (no C/C++/Python)

---

## Vision

M23-M31 made NeuralScript a production inference platform. M32-M51 has two goals:

1. **Complete the inference story**: MoE, speculative decoding, long-context, FP8 — the features every serious LLM deployment needs.
2. **Build NSL's moat**: Compile-time memory planning, roofline analysis, linear types, source-to-source AD — features that are *structurally impossible* in Python/PyTorch because they require whole-program static analysis. These make NSL not just "faster Python" but a fundamentally better tool.

### Strategic Ordering Principles

Three timing traps drove the phase ordering below:

1. **M38a Linear Types semantics must precede M32-M35 stdlib code.** If we write MoE/Ring Attention/FP8 stdlib without move semantics, introducing linear types later forces a rewrite of every function signature. M38 is split: M38a (semantic rules, behind `--linear-types` flag) ships in Phase 4 alongside the stdlib milestones; M38b (codegen optimizations, safety proofs) ships in Phase 7 when the rules have stabilized.

2. **CUDA-first for frontier features.** Ring Attention + NCCL, FP8 `mma.sync`, expert-parallel matmul are deeply NVIDIA-specific. M47 Multi-Backend ships in Phase 6 but only supports the **portable subset** (dense inference: matmul, elementwise, basic attention). FlashAttention, Ring Attention, FP8 Tensor Cores, and MoE routing kernels remain CUDA-only.

3. **Inference-first identity.** NSL's compile-time advantages (memory planning, compiled FSMs, zero-allocation) matter most for inference. M42 (KV Compression) and M44 (Constrained Decoding) are pulled forward to Phase 5; M40 (Source AD) and M43 (Pipeline Parallelism) are pushed to Phase 7.

---

## Dependency Graph

```
Phase 4: Inference Scaling, Precision & Ownership Foundations
═════════════════════════════════════════════════════════════
M38a (Linear Types — Semantics) ──> M32, M33, M34, M35 (stdlib written with ownership)
M38a ──────────────────────────────> M38b (Phase 7, codegen/safety proofs)
M32 (MoE) ───────────────────────> M50 (Sparse Tensors)
M33 (Speculative Decoding) ──────> M41 (Disaggregated Inference)
M34 (Ring Attention) ────────────> M41
M35 (FP8/Sub-Byte Quant) ───────> M32 (FP8 expert weights)

Phase 5: Compile-Time Moat & Inference Optimization
════════════════════════════════════════════════════
M36 (Memory Planning) ←── M28 (Dynamic Shapes, done), M38a (linear types for buffer reuse)
M37 (Roofline/Cost Model) ←── (standalone, uses shape info)
M42 (KV-Cache Compression) ←── M25 (PagedAttention, done)
M44 (Constrained Decoding) ←── M29 (Serve, done)

Phase 6: Deployment & Portability
═════════════════════════════════
M41 (Disaggregated Inference) ←── M29 (Serve), M33 (Spec Decode)
M39 (vmap/Auto-Batching) ←── (standalone, benefits from M49 later)
M47 (Multi-Backend, portable subset) ←── (standalone)

Phase 7: Advanced Compilation & Distribution
════════════════════════════════════════════
M38b (Linear Types — Codegen & Safety Proofs) ←── M38a
M40 (Source-to-Source AD) ←── M38b (Linear Types, full)
M43 (Pipeline Parallelism) ←── M30 (Tensor Parallelism, done)

Phase 8: Developer Experience & Ecosystem
═════════════════════════════════════════
M45 (Tensor Debugger) ←── (standalone)
M46 (Reproducibility Mode) ←── M51 (Effect System, can ship partial)
M48 (Multimodal Primitives) ←── (standalone)

Phase 9: Type System Extensions
═══════════════════════════════
M49 (Shape Algebra) ←── M28 (Dynamic Shapes, done)
M50 (Sparse Tensor Compilation) ←── M32 (MoE)
M51 (Effect System) ←── (standalone)
```

**Critical paths:**
1. **M38a → M32-M35 → M38b → M40**: Linear type semantics must be in place before writing stdlib; full codegen enables source-to-source AD.
2. **M35 → M32 → M50**: FP8 precision enables efficient MoE, which motivates sparse tensor support.
3. **M33 + M34 → M41**: Speculative decoding and ring attention feed into disaggregated inference.
4. **M51 → M46**: Effect system enables compile-time determinism proofs for reproducibility mode.
5. **M49 → M39**: Full shape algebra enables compile-time vmap transformation.

---

## Phase 4: Inference Scaling, Precision & Ownership Foundations (v0.3.0)

### M38a: Linear Types — Semantics (Ownership Checker)

**Goal:** Ship the semantic rules for linear tensor ownership behind `--linear-types` flag, so that M32-M35 stdlib code can be written with correct ownership annotations from day one.

**M38a scope (semantics only):**
- Ownership checker pass (`crates/nsl-semantic/src/ownership.rs`)
- `Ownership` enum in type system (`Owned`, `Shared`, `Borrowed`, `MutBorrowed`)
- Borrow syntax: `&x` (immutable), `&mut x` (mutable)
- `@shared` annotation for opt-in refcounting
- `@consume` parameter annotation
- Model weights `@shared` by default
- Branch/loop consumption analysis
- Closure capture semantics (move by default, `|&|` for borrow)
- `--linear-types` CLI flag (off by default — all code compiles unchanged without it)
- Error messages for use-after-move, asymmetric branch consumption, etc.

**NOT in M38a (deferred to M38b, Phase 7):**
- Refcount elision codegen
- Free-at-consumption-point emission
- M36 memory planner integration
- Autodiff tape safety proofs
- Debug-mode poison values
- Performance benchmarks

**Rationale:** The semantic rules are what M32-M35 stdlib authors need to write ownership-correct code. The codegen optimizations that exploit ownership knowledge can wait until the rules have stabilized across several milestones of real use.

**Spec:** `docs/superpowers/specs/2026-03-15-m38-linear-types-design.md` (Section 1: Language Surface + Section 2.1-2.3: Architecture + Section 5: Type System + Section 7: Testing)

---

### M32: Sparse Routing & Mixture of Experts (MoE) — COMPLETE (core), M32b pending

**Goal:** Enable MoE architectures (Mixtral, Qwen MoE, DeepSeek-V2) where a router sends each token to a subset of expert MLPs, achieving more parameters without proportional compute cost.

**M32 COMPLETE (2026-03-17):**
- `@moe(num_experts, top_k, capacity_factor, aux_loss_coeff)` decorator with semantic validation
- `moe_dispatch` compiler intrinsic with full route→scatter→gather pipeline
- Top-k gating router (softmax, capacity enforcement, token sorting)
- Scatter/gather with weighted combination (CPU fallback path)
- Auxiliary loss (CV-squared importance + load balancing)
- 4 PTX kernels (token sort histogram, scatter, expert batched GEMM, gather)
- `MoEDispatch` fusion barrier
- FFI layer (7 `#[no_mangle]` functions)
- 14 unit tests + 2 E2E tests passing

**M32b — deferred follow-up (not blocking M33-M35):**
- Expert weight matrix GEMM: wire `nsl_expert_parallel_matmul` to extract weight tensors from FixedModelArray sub-models (currently identity experts)
- Train block aux_loss injection: propagate aux_loss through autodiff tape in `compile_train_block()`
- `@expert_parallel(devices=N)`: all-to-all token exchange via M30's CollectiveBackend

**Runtime additions:** `nsl_moe_route`, `nsl_moe_scatter`, `nsl_moe_gather`, `nsl_expert_parallel_matmul`, `nsl_moe_dispatch_full`

**Spec:** `docs/superpowers/specs/2026-03-15-m32-moe-design.md`
**Plan:** `docs/superpowers/plans/2026-03-17-m32-moe-implementation.md`

---

### M33: Speculative Decoding & Tree Attention — COMPLETE (core + b)

**Goal:** Use a small draft model to guess multiple tokens, then verify them in parallel with the large model — 2-3x latency reduction for autoregressive generation.

**M33 COMPLETE (2026-03-17):**
- Rejection sampling (greedy temp=0 + stochastic temp>0 with adjusted distribution)
- Tree construction with DFS-timestamp O(1) ancestor checks + longest-path selection
- CoW page branching (block refcounting, branch, copy-on-write, cleanup)
- `@speculative` + `@medusa` decorator semantic validation + codegen extraction
- `speculative_decode()` intrinsic: takes pre-computed draft/verifier logits, returns accepted tokens
- `nsl_speculative_decode_step` FFI: full rejection sampling pipeline
- Tree attention PTX: DFS ancestor-based masking in FlashAttention Phase 2
- Scheduler `speculative_tokens` field for memory budget overhead
- 8 FFI builtins registered, `tree_mask` config in FlashAttentionConfig
- 19 unit tests + 3 E2E tests passing

**Deferred — serve block auto-speculative loop:**
- `@speculative` on `@endpoint` auto-transforming `autoregressive_decode` into draft→verify→accept loop
- Requires serve block inference pipeline maturation (tokenizer integration, KV-cache decode management)
- Core algorithms are ready; this is a codegen integration that builds on them

**Runtime additions:** `nsl_page_branch`, `nsl_page_cow_copy`, `nsl_tree_attention`, `nsl_speculative_verify`, `nsl_speculative_decode_step`

**Spec:** `docs/superpowers/specs/2026-03-15-m33-speculative-decoding-design.md`
**Plan:** `docs/superpowers/plans/2026-03-17-m33-speculative-decoding-implementation.md`

---

### M34: Context Parallelism (Ring Attention)

**Goal:** Split sequence across multiple GPUs to process million-token contexts that exceed single-GPU VRAM.

**Key components:**
- Ring attention kernel: interleave NCCL send/recv with FlashAttention tile computation
- Sequence partitioning with overlap for causal attention
- Async DMA overlap: while Tensor Cores multiply Q×K_local, DMA engine pulls K_next from neighbor GPU
- Integration with M30's `@shard` and `CollectiveBackend`
- `@context_parallel(ring_size=4)` decorator on attention functions
- Gradient checkpointing at ring segment boundaries

**Runtime additions:** `nsl_ring_attention`, `nsl_sequence_partition`, `nsl_ring_send_recv`

**Spec:** `docs/superpowers/specs/2026-03-15-m34-ring-attention-design.md`

---

### M35: FP8 Compute & Sub-Byte Quantization (AWQ/GPTQ)

**Goal:** Native FP8 Tensor Core math (2x throughput over FP16) and sub-byte weight quantization for maximum inference efficiency.

**Key components:**
- `fp8e4m3` and `fp8e5m2` as first-class compute dtypes (not just storage)
- PTX `mma.sync.aligned.m16n8k32.f16.f8.f8` for Hopper/Ada FP8 Tensor Core math
- Per-tensor and per-channel FP8 scaling with dynamic calibration
- AWQ kernels: activation-aware 4-bit weight quantization with in-register bit-unpacking during matmul
- GPTQ kernels: group-wise quantization with dequantize-in-GEMM-inner-loop
- Automatic mixed precision: type system tracks precision, compiler inserts casts
- `@fp8_compute` decorator for FP8 forward pass with FP16/FP32 master weights

**Runtime additions:** `nsl_fp8_matmul`, `nsl_fp8_cast`, `nsl_awq_matmul`, `nsl_gptq_matmul`

**Spec:** `docs/superpowers/specs/2026-03-15-m35-fp8-subbyte-quant-design.md`

---

## Phase 5: Compile-Time Moat & Inference Optimization (v0.4.0)

*Features structurally impossible in Python/PyTorch, plus inference-critical optimizations.*

### M36: Compile-Time Memory Planning

**Goal:** Eliminate runtime VRAM allocation for inference. The compiler pre-computes a memory plan, allocates once at startup, and every tensor is a pointer offset into a pre-planned slab.

**Key components:**
- Static tensor liveness analysis across the full computation graph
- Interference graph construction (tensors alive at the same time can't share memory)
- Graph coloring algorithm for memory slot assignment (like register allocation)
- `nsl build --vram-budget 8GB`: compiler *proves* the model fits or reports exactly what doesn't
- Single `cuMemAlloc` at startup for inference, zero runtime allocation
- Activation checkpointing integration: compiler inserts checkpoints to meet budget
- Compile-time peak memory report
- Integration with M38a: linear (consumed) tensors enable immediate buffer reuse in the interference graph

**Codegen changes:** New `MemoryPlanner` pass between semantic analysis and Cranelift emission. Replaces individual `checked_alloc` calls with pre-computed offsets into the memory slab.

**Spec:** `docs/superpowers/specs/2026-03-15-m36-memory-planning-design.md`

---

### M37: Compile-Time Roofline & Cost Model

**Goal:** For every operation, the compiler computes FLOPs, memory bandwidth, and arithmetic intensity, then classifies as compute-bound or memory-bound — *before you run the code*.

**Key components:**
- Per-op cost model: FLOPs and bytes-moved formulas for all 80+ tensor operations
- Target GPU spec database: theoretical peak FLOPS and bandwidth for common GPUs
- `nsl check --perf [--gpu A100]`: per-operation cost breakdown with roofline classification
- Automatic fusion suggestions: "matmul at line 42 is memory-bound, consider @fuse with following relu"
- `@perf_budget(max_tflops=150)` decorator for performance assertions
- Chrome tracing output for visualizing the cost model alongside actual runtime profiling

**Codegen changes:** New `CostModel` analysis pass. No codegen changes — purely diagnostic.

**Spec:** `docs/superpowers/specs/2026-03-15-m37-roofline-cost-model-design.md`

---

### M42: KV-Cache Compression & Eviction

**Goal:** Reduce KV-cache memory for long-context inference through quantization, eviction, and sliding window strategies.

**Key components:**
- Quantized KV-cache: store KV entries in INT8/FP8 (2-4x savings)
- Attention sink retention: keep first N tokens (attention sinks) permanently
- Sliding window with sink: `@kv_compress(window=4096, sinks=32)`
- H2O eviction: evict KV entries with lowest cumulative attention scores
- Page-level compression: integrates with M25's PagedAttention block allocator
- Per-layer compression policies (shallow layers get more aggressive compression)

**Runtime additions:** `nsl_kv_quantize`, `nsl_kv_evict`, `nsl_kv_sliding_window`

**Spec:** `docs/superpowers/specs/2026-03-15-m42-kv-compression-design.md`

---

### M44: Structured Generation / Constrained Decoding

**Goal:** Language-level support for constraining LLM output to schemas (JSON, regex, grammar). The FSM is compiled to native code — zero Python overhead.

**Key components:**
- `generate(model, prompt, schema=JsonSchema("output.schema.json"))` built-in
- Compile-time FSM construction from JSON Schema, regex, or BNF grammar
- `@grammar` annotation for inline grammar definitions
- Logit masking: FSM state → allowed token set → mask applied before softmax
- Token-level FSM stepping compiled to native code (not Python-level like Outlines)
- Integration with M29's `serve` block for serving constrained endpoints
- Batch-level FSM: different requests in a batch can have different constraints

**Codegen changes:** New `GrammarCompiler` that emits FSM transition tables as `.rodata`.

**Spec:** `docs/superpowers/specs/2026-03-15-m44-constrained-decoding-design.md`

---

## Phase 6: Deployment & Portability (v0.5.0)

### M41: Disaggregated Inference (Prefill/Decode Splitting)

**Goal:** Separate compute-bound prefill from memory-bound decode onto different hardware for optimal utilization.

**Key components:**
- `serve` block extension: `prefill_workers: 4, decode_workers: 8`
- KV-cache serialization and transfer protocol between prefill and decode nodes
- Asymmetric hardware support: prefill on compute GPUs, decode on bandwidth GPUs
- Request routing: prefill → KV transfer → decode handoff
- Integration with M33's speculative decoding (draft model on decode nodes)

**Runtime additions:** `nsl_kv_serialize`, `nsl_kv_transfer`, `nsl_disaggregated_route`

**Spec:** `docs/superpowers/specs/2026-03-15-m41-disaggregated-inference-design.md`

---

### M39: Automatic Batching (vmap)

**Goal:** Write model code for a single example; the compiler auto-vectorizes over the batch dimension.

**Key components:**
- `@vmap(batch_dim=0)` decorator on functions
- Compile-time batch dimension insertion into all operations
- Batch-invariant detection: weight matrices don't get batch dim, only activations
- Nested vmap for per-example-per-head operations
- Efficient scatter/gather for operations that genuinely need the batch dimension
- Shape error messages that distinguish batch vs. model dimensions

**Codegen changes:** AST-to-AST transformation pass that rewrites function bodies with batch dimensions before Cranelift emission.

**Spec:** `docs/superpowers/specs/2026-03-15-m39-vmap-design.md`

---

### M47: Multi-Backend Targeting (ROCm / Metal / WebGPU) — Portable Subset

**Goal:** Expand beyond CUDA-only for **dense inference workloads**. Same `kernel` keyword, portable across GPU vendors for the common operations.

**Portable subset (all backends):**
- Element-wise ops (add, sub, mul, div, neg, relu, exp, log, sqrt, abs, sign, sigmoid, tanh)
- Dense matmul (non-Tensor-Core path)
- Basic attention (naive scaled dot product, no FlashAttention)
- Softmax, LayerNorm, RMSNorm

**CUDA-only (not ported):**
- FlashAttention-2 PTX (M27)
- Ring Attention NCCL kernels (M34)
- FP8 Tensor Core `mma.sync` (M35)
- MoE radix sort / expert-parallel GEMM (M32)
- PagedAttention block-level kernels (M25)
- Reduction fusion PTX (M31)
- Epilogue fusion PTX (M31)

**Key components:**
- Backend-agnostic Kernel IR between NSL kernel AST and target-specific codegen
- ROCm/HIP backend: GCN ISA codegen, `hipMalloc`/`hipLaunchKernel` runtime
- Metal backend: MSL codegen for Apple Silicon inference
- WebGPU backend: WGSL codegen for browser deployment
- `nsl build --target rocm|metal|webgpu` flag
- Feature detection: compile-time error if kernel uses hardware-specific intrinsics not available on target
- `@target(backend)` decorator for conditional compilation

**Rationale:** Don't try to retrofit KernelIR over bespoke PTX. The frontier features (FlashAttention, Ring Attention, FP8 Tensor Cores) are deeply NVIDIA-specific and would require entirely separate implementations per backend. The portable subset covers dense inference, which is the primary use case for Metal (Apple laptops) and WebGPU (browser demos).

**Spec:** `docs/superpowers/specs/2026-03-15-m47-multi-backend-design.md`

---

## Phase 7: Advanced Compilation & Distribution (v0.6.0)

### M38b: Linear Types — Codegen & Safety Proofs

**Goal:** Now that M38a's semantic rules have been exercised across M32-M37, ship the codegen optimizations that exploit ownership knowledge.

**M38b scope:**
- Refcount elision codegen: linear tensors skip `nsl_tensor_incref`/`nsl_tensor_decref`
- Free-at-consumption-point emission: `nsl_tensor_free` emitted at move site, not scope exit
- Autodiff tape safety proofs: ownership checker validates `saved_a`/`saved_b` liveness
- Debug-mode poison values: zeroed slots after move for null-pointer-on-reuse detection
- M36 memory planner integration: linear (consumed) tensors → immediate buffer reuse
- Performance benchmarks: measure refcount ops eliminated on typical forward passes

**Spec:** `docs/superpowers/specs/2026-03-15-m38-linear-types-design.md` (Section 3: Runtime + Section 4: Codegen + Section 6: Autodiff Safety)

---

### M40: Source-to-Source Automatic Differentiation

**Goal:** Replace tape-based (runtime) autodiff with compile-time source transformation. Eliminate tape overhead, enable dead gradient elimination, and support higher-order derivatives.

**Key components:**
- Forward-mode AD for Jacobian-vector products
- Reverse-mode AD via source transformation (compiler generates backward functions)
- Dead gradient elimination: prune unused gradient computations at compile time
- Cross-boundary fusion: fuse operations across forward/backward boundary
- Higher-order derivatives: `grad { grad { f(x) } }` for Hessians, meta-learning (MAML)
- Fallback to tape-based AD for dynamic control flow (while loops with data-dependent termination)
- Migration path: existing `grad` keyword semantics preserved, compiler chooses strategy

**Codegen changes:** New `DifferentiationPass` that transforms AST before codegen. Requires M38b's full linear types to prove safety of the transformation.

**Spec:** `docs/superpowers/specs/2026-03-15-m40-source-ad-design.md`

---

### M43: Pipeline Parallelism & Distributed Training

**Goal:** Split model layers across GPUs for training. Combined with M30's tensor parallelism, enables full 3D parallelism (data + tensor + pipeline).

**Key components:**
- `@pipeline(stages=4)` decorator on model definitions
- Automatic layer partitioning with balanced compute per stage
- 1F1B (one-forward-one-backward) micro-batch schedule for minimal pipeline bubble
- ZeRO-style optimizer state sharding across data-parallel workers
- Gradient accumulation: `train { gradient_accumulation_steps: 4 }`
- Activation checkpointing at pipeline stage boundaries
- `train { distribute: "dp=2, tp=4, pp=4" }` for 3D parallelism config

**Runtime additions:** `nsl_pipeline_send`, `nsl_pipeline_recv`, `nsl_zero_scatter`, `nsl_zero_gather`

**Spec:** `docs/superpowers/specs/2026-03-15-m43-pipeline-parallelism-design.md`

---

## Phase 8: Developer Experience & Ecosystem (v0.7.0)

### M45: Time-Travel Tensor Debugger

**Goal:** Record every tensor operation with statistics, enabling forward/backward stepping through the computation graph and automatic NaN/Inf root cause identification.

**Key components:**
- `nsl run --trace`: records op name, shapes, dtypes, min/max/mean/std per tensor
- Compact binary trace format (~100 bytes per op)
- `nsl debug <trace-file>`: interactive TUI with forward/backward stepping
- NaN/Inf sentinel: breaks on first operation producing NaN/Inf, shows full computation chain
- Compile-time NaN risk analysis: warns about `log(x)` where `x` could be 0, `1/x` division
- Trace diffing: compare two runs to find where outputs diverge
- Integration with Chrome tracing for timeline visualization

**Runtime additions:** `nsl_trace_op`, `nsl_trace_tensor_stats`, `nsl_trace_dump`

**Spec:** `docs/superpowers/specs/2026-03-15-m45-tensor-debugger-design.md`

---

### M46: Reproducibility Mode

**Goal:** Compile-time guarantee of deterministic execution. The compiler proves no non-deterministic operations exist.

**Key components:**
- `nsl run --deterministic`: enforces deterministic execution mode
- Compile-time non-determinism detection: warns about atomicAdd reductions, non-deterministic cuDNN algorithms
- Deterministic alternatives: compiler auto-selects deterministic GPU kernels when available
- RNG seed tracking: compiler ensures all random operations use explicit seeds
- Checkpoint fingerprinting: hash of computation graph + seeds + data order
- Bit-exact reproducibility across runs (same hardware)
- Requires M51's effect system for `Random` effect tracking

**Codegen changes:** `DeterminismChecker` pass that validates all operations have deterministic implementations.

**Spec:** `docs/superpowers/specs/2026-03-15-m46-reproducibility-design.md`

---

### M48: Native Multimodal Primitives

**Goal:** First-class language constructs for vision, audio, and cross-modal attention.

**Key components:**
- `vision_encoder` model block: ViT patch embedding, convolution stems
- `audio_encoder` model block: mel spectrogram, wav2vec-style feature extraction
- `cross_attend(queries, keys, values)` primitive with compile-time cross-modal shape verification
- Named dimensions for multimodal: `Tensor<[batch, patches, channels]>` vs `Tensor<[batch, seq, hidden]>`
- Image preprocessing builtins: `resize`, `normalize`, `patch_embed`
- Audio preprocessing builtins: `mel_spectrogram`, `stft`, `resample`
- `@multimodal` decorator for models that accept multiple input modalities

**Runtime additions:** `nsl_patch_embed`, `nsl_mel_spectrogram`, `nsl_cross_attention`

**Spec:** `docs/superpowers/specs/2026-03-15-m48-multimodal-design.md`

---

## Phase 9: Type System Extensions (v0.8.0)

### M49: Compile-Time Shape Algebra & Dependent Dimensions

**Goal:** Extend NSL's existing `DimExpr` into a full symbolic solver that propagates shapes through operations and proves reshape/split/concat safety at compile time.

**Key components:**
- Symbolic propagation: `reshape(Tensor<[B, S*D]>, [B, S, D])` — compiler solves `S*D = S*D`
- Divisibility proofs: `x.reshape([B, S, D//2, 2])` — compiler proves `D` is even (or rejects)
- Conditional shapes: compiler tracks shape through both branches of `if`/`match`
- Shape error messages with algebra: "got D=768 vs D_k=64*11=704"
- Bounded dimension integration: `SeqLen < 4096` combined with algebra for range proofs
- `@shape_assert` decorator for user-specified shape invariants

**Semantic changes:** Extend `DimExpr` with a constraint solver (SMT-lite). Integrate with M28's `SymbolicDimTracker`.

**Spec:** `docs/superpowers/specs/2026-03-15-m49-shape-algebra-design.md`

---

### M50: Sparse Tensor Compilation

**Goal:** Type-driven kernel dispatch for sparse tensors. NSL's type system already declares `Sparse<format>` — this wires it to actual kernels.

**Key components:**
- Format-aware kernel dispatch: `matmul(dense, sparse_csr)` → CSR SpMM kernel
- Sparsity-preserving type tracking: compiler knows which operations preserve sparsity
- Structured sparsity: `@sparse(pattern="2:4")` for Ampere Sparse Tensor Core support
- Compile-time sparsity pattern verification for structured formats
- Conversion ops: `to_sparse(format="csr")`, `to_dense()` with cost warnings
- MoE integration: sparse routing matrices use sparse tensor operations natively

**Runtime additions:** `nsl_sparse_create`, `nsl_sparse_spmm`, `nsl_sparse_spmv`, `nsl_sparse_to_dense`

**Spec:** `docs/superpowers/specs/2026-03-15-m50-sparse-tensors-design.md`

---

### M51: Effect System & Safe Parallelism

**Goal:** Track computational effects (IO, Random, Mutation, Communication) through the type system. Enable compile-time proofs about parallelism safety, determinism, and checkpointability.

**Key components:**
- Effect annotations: `IO`, `Random`, `Mutation`, `Communication`, `Pure`
- `@pure` verification: compiler proves function has no effects
- `@deterministic`: no `Random` or non-deterministic `Communication` effects
- Safe gradient checkpointing: compiler proves checkpointed functions are `Pure` (recomputation-safe)
- Safe parallelism: `@pure` functions can be auto-parallelized without data races
- Communication effect pairing: compiler verifies all-reduce/all-gather are properly paired across ranks
- Effect inference: compiler infers effects from function body (no annotation needed for most cases)

**Semantic changes:** New `EffectChecker` pass. Effects propagate through call graph.

**Spec:** `docs/superpowers/specs/2026-03-15-m51-effect-system-design.md`

---

## Release Plan

| Version | Milestones | Theme | Notes |
|---------|-----------|-------|-------|
| v0.3.0 | M38a, M32-M35 | Inference Scaling, Precision & Ownership Foundations | M38a ships semantics only (behind `--linear-types` flag) |
| v0.4.0 | M36, M37, M42, M44 | Compile-Time Moat & Inference Optimization | NSL's "moat" features + inference DX |
| v0.5.0 | M41, M39, M47 | Deployment & Portability | M47 = portable subset only (dense inference) |
| v0.6.0 | M38b, M40, M43 | Advanced Compilation & Distribution | M38b completes linear types; training at scale |
| v0.7.0 | M45, M46, M48 | Developer Experience & Ecosystem | — |
| v0.8.0 | M49, M50, M51 | Type System Extensions | — |

---

## Parallelization Opportunities

Within each phase, some milestones can be developed in parallel:

- **Phase 4:** M38a should start first (or at least concurrently). M32, M33, M34, M35 are independent of each other. M35 enables FP8 expert weights for M32 but can overlap.
- **Phase 5:** M36 and M37 are independent. M42 and M44 are independent. All four can be parallelized.
- **Phase 6:** M41, M39, M47 are all independent.
- **Phase 7:** M38b must come first. M40 depends on M38b. M43 is independent.
- **Phase 8:** M45, M46, M48 are all independent (M46 benefits from M51 but can ship partial).
- **Phase 9:** M49, M50, M51 are independent.

---

## Spec Documents

Each milestone has a detailed design spec:

| Milestone | Spec Path |
|-----------|-----------|
| M32 | `docs/superpowers/specs/2026-03-15-m32-moe-design.md` |
| M33 | `docs/superpowers/specs/2026-03-15-m33-speculative-decoding-design.md` |
| M34 | `docs/superpowers/specs/2026-03-15-m34-ring-attention-design.md` |
| M35 | `docs/superpowers/specs/2026-03-15-m35-fp8-subbyte-quant-design.md` |
| M36 | `docs/superpowers/specs/2026-03-15-m36-memory-planning-design.md` |
| M37 | `docs/superpowers/specs/2026-03-15-m37-roofline-cost-model-design.md` |
| M38 | `docs/superpowers/specs/2026-03-15-m38-linear-types-design.md` |
| M39 | `docs/superpowers/specs/2026-03-15-m39-vmap-design.md` |
| M40 | `docs/superpowers/specs/2026-03-15-m40-source-ad-design.md` |
| M41 | `docs/superpowers/specs/2026-03-15-m41-disaggregated-inference-design.md` |
| M42 | `docs/superpowers/specs/2026-03-15-m42-kv-compression-design.md` |
| M43 | `docs/superpowers/specs/2026-03-15-m43-pipeline-parallelism-design.md` |
| M44 | `docs/superpowers/specs/2026-03-15-m44-constrained-decoding-design.md` |
| M45 | `docs/superpowers/specs/2026-03-15-m45-tensor-debugger-design.md` |
| M46 | `docs/superpowers/specs/2026-03-15-m46-reproducibility-design.md` |
| M47 | `docs/superpowers/specs/2026-03-15-m47-multi-backend-design.md` |
| M48 | `docs/superpowers/specs/2026-03-15-m48-multimodal-design.md` |
| M49 | `docs/superpowers/specs/2026-03-15-m49-shape-algebra-design.md` |
| M50 | `docs/superpowers/specs/2026-03-15-m50-sparse-tensors-design.md` |
| M51 | `docs/superpowers/specs/2026-03-15-m51-effect-system-design.md` |
