<!-- owner: @bwiemz -->

# Glossary

Every acronym, keyword, and decorator you will see in NSL code, specs, or PRs. One-line definitions with links to primary references. For language-level constructs, follow the link to the authoritative [`spec/`](../../spec/) file.

---

## Subsystems

### <a id="awq"></a>AWQ — Activation-aware Weight Quantization
Build-time per-channel weight quantization that scales by observed activation magnitudes; produces INT4/INT8 weights with minimal perplexity loss. See [`spec/06-quantization.nsl.md`](../../spec/06-quantization.nsl.md); implementation in `crates/nsl-codegen/src/calibration/`.

### <a id="ccr"></a>CCR — Common-kernel Combination Rewriting
IR pass that merges semantically identical kernel invocations (same op, same shape, same arguments) into a single dispatch before FASE and WGGO fire. See [CCR.pdf](../research/CCR.pdf); fires as the first optimization pass in `compile_quant_block`.

### <a id="cep"></a>CEP — Compilation-Evaluated Pruning
Compile-time head-pruning pass: the oracle, importance scorer, model rewriter, and greedy search are all applied before the binary is emitted — pruned heads generate no code at all. Implementation: `crates/nsl-codegen/src/cep.rs`; decisions surfaced via `nsl profile --explain-wggo`.

### <a id="cesh"></a>CESH — Compiler-Evaluated Speculative Hypothesis
Research paper describing NSL's speculative decoding approach: hypothesis generation and verification are modeled as a single compiler-evaluated IR transformation. See [CESH.pdf](../research/CESH.pdf).

### <a id="cfa"></a>CFA — Compiler-Fused Attention (research)
Research document covering the compiler's strategy for fusing multi-head attention with adjacent elementwise and normalization ops. See [NSL-CFA-Research.md.pdf](../research/NSL-CFA-Research.md.pdf).

### <a id="cfie"></a>CFIE — Compiler-Fused Inference Engine
NSL's inference-serving orchestration layer: KV plan, fused sampling, speculative decode, persistent KV cache, and grammar-constrained generation all compose into one codegen pass. See [CFIE.pdf](../research/CFIE.pdf); implementation in `crates/nsl-codegen/src/cfie.rs`.

### <a id="cftp"></a>CFTP — Compiler-Fused Training Pipeline
Research framework covering NSL's full training pipeline as a compile-time transformation: batching, packing, gradient accumulation, and optimizer dispatch fused into a single low-level loop. See [CFTP.pdf](../research/CFTP.pdf); PCA (§2) and FASE (§1) are the shipped sub-systems.

### <a id="cgac"></a>CGAC — Compiler-Guided Activation Checkpointing
Research document describing NSL's compile-time strategy for inserting `@checkpoint` recomputation points to trade compute for memory, guided by the memory planner's interference graph. See [CGAC.pdf](../research/CGAC.pdf).

### <a id="cidp"></a>CIDP — Compiler-Integrated Data Pipeline
Research document covering NSL's data-loading and tokenization pipeline as a compiler-visible IR construct, enabling fusion with the training loop. See [NSL-CIDP-Research.md.pdf](../research/NSL-CIDP-Research.md.pdf).

### <a id="ckvq"></a>CKVQ — Compiler-aware KV-cache Quantization
Research paper on compile-time KV-cache quantization: the compiler annotates KV tensors with dtype/precision decisions at build time rather than selecting them at runtime. See [CKVQ.pdf](../research/CKVQ.pdf).

### <a id="cpdt"></a>CPDT — Compile-time Parallelism & Distributed Training
Pass that plans ZeRO-style tensor sharding, collective-comm scheduling, mixed precision, and expert placement across a `ClusterSpec` before binary emission. See [CPDT Research.pdf](<../research/CPDT Research.pdf>); design specs in [`docs/superpowers/specs/`](../superpowers/specs/) (search `cpdt`); implementation in `crates/nsl-codegen/src/cpdt.rs`.

### <a id="cpkd"></a>CPKD — Compiler-Planned Knowledge Distillation
Research document covering compile-time wiring of teacher and student models so the KL-divergence distillation loss and gradient flow are lowered as a single fused Wengert list. See [CPKD.pdf](../research/CPKD.pdf).

### <a id="csha"></a>CSHA — Compiler-Synthesized Holistic Attention
Compile-time pass that fuses RMSNorm + Q/K/V projection (via m16n8k16 MMA) + RoPE rotation + FlashAttention-2 into a single PTX kernel, eliminating intermediate HBM traffic. See [NSL-CSHA-Research.PDF](../research/NSL-CSHA-Research.PDF); design specs in [`docs/superpowers/specs/`](../superpowers/specs/) (search `csha`); implementation in `crates/nsl-codegen/src/csha.rs`.

### <a id="css"></a>CSS — Compiler-Scheduled Sparsity
Research document on compile-time sparsity scheduling: the compiler decides where to insert and exploit structured sparsity patterns (block-sparse, N:M) at build time. See [CSS.pdf](../research/CSS.pdf).

### <a id="fase"></a>FASE — Fused Accumulation-Step Elimination
Training optimization pass that defers gradient accumulation into a fused per-parameter loop (Deferred mode) or buffers it explicitly (FullBuffer mode), eliminating redundant optimizer-step dispatch. See [`docs/superpowers/specs/`](../superpowers/specs/) (search `fase`); implementation in `crates/nsl-codegen/src/fase.rs`.

### <a id="pca"></a>PCA — Packed Causal Attention
FlashAttention-2 kernel variant that packs multiple variable-length sequences into a single tile pass, using per-segment masking to preserve causal isolation. Defined in CFTP §2. See [`docs/superpowers/specs/2026-04-18-pca-tier-a-design.md`](../superpowers/specs/2026-04-18-pca-tier-a-design.md); implementation in `crates/nsl-codegen/src/pca_segment.rs`.

### <a id="wggo"></a>WGGO — Wengert Graph Global Optimization
Multi-level ILP/DP pass that co-optimizes CEP head pruning, CSHA fusion level, WRGA adapter rank, and CPDT shard factor across all layers jointly, producing an `AppliedPlan` that downstream consumers consume. See [NSL-WGGO-Research.md.pdf](../research/NSL-WGGO-Research.md.pdf); implementation in `crates/nsl-codegen/src/wggo.rs`.

### <a id="wrga"></a>WRGA — Wengert-Pruned Roofline-Guided Adaptation
Pass that decides per-layer LoRA/IA³/GatedLoRA adapter ranks by combining Wengert-graph pruning, roofline bandwidth analysis, and spectral rank scoring, then emits fused MMA PTX for the adapter forward (and optionally backward) pass. See [NSL-WRGA-Research.PDF](../research/NSL-WRGA-Research.PDF); design specs in [`docs/superpowers/specs/`](../superpowers/specs/) (search `wrga`); implementation in `crates/nsl-codegen/src/wrga.rs`.

---

## Hardware / GPU terms

### <a id="cublas"></a>cuBLAS
NVIDIA's CUDA BLAS library. NSL does not link cuBLAS — it emits its own MMA-based matmul PTX so the binary stays zero-dependency. Mentioned in design docs as a correctness reference baseline.

### <a id="cudarc"></a>cudarc
Rust crate (version 0.19) providing safe wrappers around the CUDA driver API. NSL uses it for device-memory allocation, PTX module loading, and kernel launch. See `Cargo.toml` feature flags `dynamic-linking` + `cuda-version-from-build-system`.

### <a id="fa"></a>FA / FA-2 — FlashAttention / FlashAttention-2
Tiled online-softmax attention algorithm (Dao et al.) that keeps Q, K, V tiles in SRAM to avoid full HBM materialisation of the N×N attention matrix. NSL's FA-2 scalar v2 emitter lives in `crates/nsl-codegen/src/flash_attention_v2/`; activated via the `@flash_attention` decorator or the `@csha` decorator chain. See [NSL-CFA-Research.md.pdf](../research/NSL-CFA-Research.md.pdf) for the NSL-specific fusion research.

### <a id="fp8"></a>FP8
8-bit floating-point format (E4M3 or E5M2) supported on H100/Hopper and RTX 4090+. NSL uses FP8 for V tensors in CSHA Tier D (per-head mixed precision); E4M3 for Q/K (range-critical) and E5M2 for V (gradient-safe). See [`spec/06-quantization.nsl.md`](../../spec/06-quantization.nsl.md).

### <a id="gqa"></a>GQA — Grouped-Query Attention
Attention variant where multiple query heads share a single K/V head, reducing KV-cache memory. NSL's FA-2 emitter handles GQA via `source_ad.rs` expand-backward `ReduceToShape` for KV head gradients.

### <a id="gptq"></a>GPTQ — Generative Pre-Trained Quantization
Post-training weight quantization algorithm (Frantar et al.) that uses second-order Hessian information to minimize quantization error layer by layer. Supported via `quant` block in NSL. See [`spec/06-quantization.nsl.md`](../../spec/06-quantization.nsl.md).

### <a id="int4"></a>INT4 / INT8
Integer quantization dtypes. `INT4` packs two 4-bit values per byte; `INT8` is one byte per value. Both are supported in NSL's `quant` block with per-channel or per-group granularity. See [`spec/06-quantization.nsl.md`](../../spec/06-quantization.nsl.md).

### <a id="kir"></a>KIR — Kernel IR
NSL's flat SSA-form intermediate representation for GPU kernels, defined in `crates/nsl-codegen/src/kernel_ir.rs`. KIR sits between the `kernel` block AST and backend PTX/WGSL/AMDGPU emission. From M47 onward, KIR is the portable abstraction that non-CUDA backends target.

### <a id="mma"></a>MMA — Matrix Multiply-Accumulate
PTX instruction family (`mma.sync.aligned.m16n8k16`, `wgmma.mma_async`, etc.) that performs tile-level matrix multiplication directly in registers. NSL uses `mma.sync.m16n8k16` (Ampere, sm_80) for WRGA fused adapter forward and CSHA projection PTX. Primitives live in `crates/nsl-codegen/src/matmul_mma.rs`.

### <a id="ptx"></a>PTX — Parallel Thread Execution
NVIDIA's virtual ISA for GPU kernels. NSL emits PTX text directly (no LLVM or nvcc); PTX is then assembled to a `.cubin` by ptxas (embedded in the CUDA toolkit). Key rule: all PTX comments must be ASCII-only — non-ASCII characters pass offline ptxas but trigger `CUDA_ERROR_INVALID_PTX` in cudarc at runtime.

### <a id="rope"></a>RoPE — Rotary Position Embedding
Positional encoding scheme (Su et al.) that applies a complex rotation to Q and K vectors in-register, making attention scores relative-position-aware without adding learned parameters. NSL's FA-2 emitter has a `rope_q` flag; CSHA Tier A includes a fused RoPE epilogue PTX pass.

### <a id="tma"></a>TMA — Tensor Memory Accelerator (Hopper)
Hopper-generation (sm_90) hardware unit that performs bulk async copies between global and shared memory using `cp.async.bulk.tensor` instructions. NSL's Hopper path in `flash_attention.rs` has TMA plumbing; the full wgmma-based Hopper kernel is not yet production-complete.

### <a id="wcet"></a>WCET — Worst-Case Execution Time
Compile-time upper bound on kernel execution time, required for hard real-time / robotics workloads (M53). NSL's `nsl check --wcet` pass in `crates/nsl-codegen/src/wcet.rs` models loop trip counts and memory latencies to produce a provable bound.

### <a id="zk"></a>ZK — Zero-Knowledge (inference)
Zero-knowledge proof system for ML inference (M55): the prover runs the model and produces a proof that the output was computed correctly without revealing weights. NSL supports a `nsl build --zk` path that compiles an arithmetic-circuit representation alongside the native binary. Implementation scaffolded in `crates/nsl-codegen/src/zk/`.

---

## NSL keywords

### <a id="kw-const"></a>`const`
Immutable binding; value must be a compile-time literal or constant expression. Full reference: [`spec/01-syntax-fundamentals.nsl.md`](../../spec/01-syntax-fundamentals.nsl.md).

### <a id="kw-datatype"></a>`datatype`
Block keyword (M23) for defining custom numeric datatypes with user-specified packing, arithmetic, and PTX escape hatches; enables BYOD (Bring Your Own Dtype) quantization formats. Full reference: [`spec/06-quantization.nsl.md`](../../spec/06-quantization.nsl.md).

### <a id="kw-fn"></a>`fn`
Function definition keyword. Full reference: [`spec/01-syntax-fundamentals.nsl.md`](../../spec/01-syntax-fundamentals.nsl.md).

### <a id="kw-grad"></a>`grad`
Keyword (not a library call) that initiates a gradient computation; the compiler inserts tape start/stop and backward-pass emission automatically. `grad` is reserved — optimizer stdlib parameters must use `gradient`. Full reference: [`spec/03-automatic-differentiation.nsl.md`](../../spec/03-automatic-differentiation.nsl.md).

### <a id="kw-kernel"></a>`kernel`
Block keyword for writing custom GPU operations; compiles to PTX at build time via `crates/nsl-codegen/src/kernel.rs`. Full reference: [`spec/09-hardware-abstraction.nsl.md`](../../spec/09-hardware-abstraction.nsl.md).

### <a id="kw-let"></a>`let`
Mutable binding. Full reference: [`spec/01-syntax-fundamentals.nsl.md`](../../spec/01-syntax-fundamentals.nsl.md).

### <a id="kw-model"></a>`model`
Block keyword for defining a model (analogous to a class, but keyword-level so the compiler can track weight tensors and generate serialization automatically). `model` is reserved — parser rejects `model=m` as an identifier. Full reference: [`spec/04-model-definition.nsl.md`](../../spec/04-model-definition.nsl.md).

### <a id="kw-quant"></a>`quant`
Block keyword for declarative quantization configuration (INT4, INT8, FP8, GPTQ, AWQ, per-channel/group granularity). Full reference: [`spec/06-quantization.nsl.md`](../../spec/06-quantization.nsl.md).

### <a id="kw-serve"></a>`serve`
Block keyword (M29) for the continuous-batching serving engine: declares prefill/decode chunking, preemption policy, and KV-cache management as compile-time configuration. Full reference: [`spec/09-hardware-abstraction.nsl.md`](../../spec/09-hardware-abstraction.nsl.md).

### <a id="kw-train"></a>`train`
Block keyword for the declarative training DSL; generates an epoch loop with implicit tape_start/backward/stop and per-parameter optimizer dispatch. Full reference: [`spec/05-training-loop.nsl.md`](../../spec/05-training-loop.nsl.md).

---

## Decorators

### <a id="dec-autotune"></a>`@autotune`
Triggers build-time tuning of tile sizes, warp counts, and SMEM allocation for a `kernel` block; stores the winning config in the binary's autotune table. See [`spec/09-hardware-abstraction.nsl.md`](../../spec/09-hardware-abstraction.nsl.md).

### <a id="dec-backward"></a>`@backward`
Marks a function as a hand-written backward pass that overrides the compiler's source-AD-generated adjoint for a particular op. Full reference: [`spec/03-automatic-differentiation.nsl.md`](../../spec/03-automatic-differentiation.nsl.md).

### <a id="dec-checkpoint"></a>`@checkpoint`
Marks a function boundary where activations are recomputed on the backward pass instead of stored; reduces peak memory at the cost of one extra forward pass through the marked region. Full reference: [`spec/03-automatic-differentiation.nsl.md`](../../spec/03-automatic-differentiation.nsl.md).

### <a id="dec-cpdt"></a>`@cpdt`
Attaches CPDT sharding and distributed-training configuration to a model; `@cpdt(weight_aware=false)` opts out of weight-aware calibration. Enforces one-`@cpdt`-per-program semantic. See [Glossary#cpdt](#cpdt); design spec `2026-04-18-cpdt-weight-aware-phase1-design.md`.

### <a id="dec-csha"></a>`@csha`
Enables CSHA fusion for a `@flash_attention`-decorated function; accepts `level`, `target`, and `disable` parameters. See [Glossary#csha](#csha).

### <a id="dec-export"></a>`@export`
Marks a function for PyTorch FFI export (M62), generating a C-ABI wrapper and Python binding. Shipped PR #45 (2026-04-15). See [Adding-a-Language-Feature](Adding-a-Language-Feature.md) for end-to-end walkthrough; design spec `2026-04-15-m62-export-decorator-design.md`.

### <a id="dec-flash-attention"></a>`@flash_attention`
Signals to the compiler that the decorated function implements multi-head attention and should be lowered to the FA-2 PTX emitter; defaults to `causal=true`. See [Glossary#fa](#fa); feedback file `feedback_fa_decorator_causal_default.md` for the `row_sum=[1,2,...,N]` invariant.

### <a id="dec-freeze"></a>`@freeze`
Marks model parameters as frozen during training; the compiler removes their VarIds from `backward_live` so no adjoint is emitted for them. Used in conjunction with WRGA adapter injection.

### <a id="dec-fuse"></a>`@fuse`
Requests elementwise fusion of the decorated function with its producer/consumer ops (M26/M31); the compiler attempts to merge them into a single kernel dispatch via the epilogue fusion pass in `epilogue_fusion.rs`.

### <a id="dec-inspect"></a>`@inspect`
Compiler-native debugger decorator (Dev Tools Phase 1) that captures tensor state at the decorated call site and writes an NSLI-format dump, readable via `nsl run --monitor`. Implementation: `crates/nsl-codegen/src/inspect/`.

### <a id="dec-no-grad"></a>`@no_grad`
Disables tape recording inside the decorated function; all operations are forward-only. Equivalent to PyTorch's `torch.no_grad()` but enforced at compile time. Full reference: [`spec/03-automatic-differentiation.nsl.md`](../../spec/03-automatic-differentiation.nsl.md).

### <a id="dec-shard"></a>`@shard`
Marks a tensor as sharded across the GPU world (M30); the compiler inserts NCCL collective ops at the shard boundaries. Full reference: [`spec/09-hardware-abstraction.nsl.md`](../../spec/09-hardware-abstraction.nsl.md).

### <a id="dec-tie-weights"></a>`@tie_weights`
Declares that two model fields share the same underlying weight tensor (e.g., embedding and output projection); the compiler emits a single weight slot and generates tied-gradient accumulation. Full reference: [`spec/04-model-definition.nsl.md`](../../spec/04-model-definition.nsl.md).

### <a id="dec-wrga"></a>`@wrga`
Attaches WRGA adapter configuration to a model, specifying adapter type (LoRA/IA³/GatedLoRA), rank bounds, and target layers. Used alongside `@freeze` and `@adapter`. See [Glossary#wrga](#wrga).

---

## Milestone prefix convention

`M<N>` where `N` ∈ [9, 62]. Each milestone maps to a phase and version:

- **M9–M20** → v0.1.0 (language foundation, GPU, interop)
- **M23–M31** → v0.2.0 (production inference)
- **M32–M51** → v0.3.0–v0.8.0 (scaling, moat features, deployment)
- **M52–M62** → v0.9.0–v1.2.0 (weight intelligence, safety, frontier scale)

See [Roadmap](Roadmap.md) for the full phase → version → milestone mapping.

---

## Primary references (PDFs in repo root)

| Acronym | PDF |
|---------|-----|
| CSHA    | [NSL-CSHA-Research.PDF](../research/NSL-CSHA-Research.PDF) |
| WRGA    | [NSL-WRGA-Research.PDF](../research/NSL-WRGA-Research.PDF) |
| WGGO    | [NSL-WGGO-Research.md.pdf](../research/NSL-WGGO-Research.md.pdf) |
| CPDT    | [CPDT Research.pdf](<../research/CPDT Research.pdf>) |
| CFTP    | [CFTP.pdf](../research/CFTP.pdf) |
| CPKD    | [CPKD.pdf](../research/CPKD.pdf) |
| CCR     | [CCR.pdf](../research/CCR.pdf) |
| CEP     | [CEP.pdf](../research/CEP.pdf) |
| CESH    | [CESH.pdf](../research/CESH.pdf) |
| CFIE    | [CFIE.pdf](../research/CFIE.pdf) |
| CGAC    | [CGAC.pdf](../research/CGAC.pdf) |
| CKVQ    | [CKVQ.pdf](../research/CKVQ.pdf) |
| CSS     | [CSS.pdf](../research/CSS.pdf) |
| CFA     | [NSL-CFA-Research.md.pdf](../research/NSL-CFA-Research.md.pdf) |
| CIDP    | [NSL-CIDP-Research.md.pdf](../research/NSL-CIDP-Research.md.pdf) |

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21. If the crate graph or pass order in this page no longer matches reality, open an issue tagged `docs-rot`.*
