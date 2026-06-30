# Changelog

All notable changes to NeuralScript will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added — Architecture hardening (stable/experimental boundaries, ABI versioning)

- `STATUS.md` — single source of truth tiering every subsystem as Stable / Beta /
  Experimental, with the per-tier test expectations.
- `docs/hardware/` — tested-on matrix plus `cuda_status.md` and `fpga_status.md`,
  making GPU/FPGA claims traceable (Validated / Built / Analysis-only) to actual
  evidence instead of aspirational support.
- `docs/abi/README.md` — runtime C-ABI contract: versioning policy and the
  per-symbol FFI safety checklist.
- Runtime C-ABI versioning: `nsl_runtime::c_api::{NSL_ABI_VERSION_MAJOR,
  NSL_ABI_VERSION_MINOR}` constants and the `nsl_abi_version()` exported fn
  (packed `(major<<16)|minor`). Generated C headers now emit matching
  `NSL_ABI_VERSION_*` macros (pinned to the runtime constants) and the
  `nsl_abi_version()` prototype, so hosts can detect runtime/header skew.
- Golden ABI-layout test pinning `NslTensorDesc` to 48 bytes / 8-byte align /
  fixed field offsets (`nsl_tensor_desc_abi_layout_is_pinned`).
- `CONTRIBUTING.md` — "Review gates" section formalizing the stable-vs-
  experimental, FFI, hardware-claim, config-sprawl, and clippy-suppression
  boundaries; test requirements split into required-on-PR / nightly / research.

### Changed

- `SECURITY.md` — corrected the supported-versions table (now `main` + 0.9.x)
  and enumerated the highest-risk areas (C ABI, dlopen, model/weight parsers,
  path handling, CUDA launch, generated-code execution, compiler DoS).
- `README.md` — replaced the inaccurate "no runtime" claim with "no Python
  interpreter / no GIL; programs link a small native runtime", and added a
  pre-1.0 maturity pointer to `STATUS.md`.

### Fixed

- `test-onnx-rt` CI job: install the `rustfmt` toolchain component, which
  `bindgen` requires when generating bindings in `tools/verify-ort-vendoring.sh`
  (the pinned 1.95.0 toolchain ships without it, failing the job at the
  vendoring step independently of ONNX itself).

### Added — CSHA Tier B.2 backward Phase 2 (foundation + dQ-kernel emitter)

- `flash_attention_v2::tier_b2::backward::d_prepass::synthesize_d_prepass` — D pre-pass kernel emitter (row-per-lane schedule: 32 lanes × 1 row each, sequential over `head_dim`; no inter-lane reduction; no SMEM; sm_80+; computes `D[b,h,q] = rowsum(dO * O)`). Spec §3.3's original butterfly-reduction schedule was replaced after the first GPU launch revealed a row/col conflation bug — both schedules are HBM-bandwidth-bound at canonical sizes, and the row-per-lane schedule avoids the bug class entirely.
- `flash_attention_v2::tier_b2::backward::dq::synthesize_dq_kernel` — dQ-kernel emitter (~700 LOC), producer-consumer warp specialization, register-resident dQ accumulator across kv-inner loop, **no atomicAdd**. Inner-loop MAC chain: S = QK^T → P recompute → dP = dO·V^T → dS = P·(dP - D) → col-major K re-stage band (Path A) → dQ_acc += dS @ K.
- `flash_attention_v2::tier_b2::register_budget_backward` — `BackwardKernel` enum (DPrePass | DQ | DKDV), `count_registers_backward`, `predict_fallback` planner helper covering BOTH SMEM-pressure (hd=128) and register-pressure (hd=256) cases.
- `flash_attention_v2::smem_layout::tier_b2_dq_*_offset` accessors including the new `tier_b2_dq_k_colmajor_offset` Path A re-stage band, plus `tier_b2_effective_bq`/`tier_b2_effective_bkv` per-hd fallback schedule and `tier_b2_dkdv_*` stubs for Phase 3.
- `matmul_mma::emit_load_b_fragment_smem` parameter renamed `row_stride_bytes` → `col_stride_bytes` (was misnamed for B.1's actual use as the column stride between adjacent n-axis positions). The Task 2 `load_transposed: bool` extension was reverted after V-B2-5 verification found it architecturally unsound (commit `275d849d`).
- `nsl-test` crate (new workspace member) with `nsl_test::diagnostic_mode::{DSource, compute_d_for_test}` — permanent test utility for backward-kernel localizability (swap CPU-D in for B.2-pre-pass-D to bisect failures). Phase 3 dK/dV-kernel and future-milestone backward work inherit the primitive.
- `crates/nsl-codegen/tests/tier_b2_no_atomic_in_dq.rs` — Rust-level PTX-parse invariant test (CPU-only, runs every commit; asserts dQ-kernel emits zero `atom.*` instructions per spec §7.2).
- `crates/nsl-codegen/tests/tier_b2_dq_k_colmajor_lane_mapping.rs` — Spec §5.5 institutional pin: lane-mapping byte-pattern test for the col-major K re-stage band.
- `crates/nsl-codegen/tests/tier_b2_dq_kernel_cpu_reference.rs` — Layer-1 dQ tests (Test 1: D pre-pass standalone; Test 2: dQ smoke at canonical; Test 3: dQ head_dim sweep across {32, 64, 128}). All `#[ignore]` + `feature="cuda"` — manual GPU validation gates Phase 2 closure.
- `crates/nsl-test/tests/diagnostic_mode_localizes_d_bug.rs` — Spec §7.3 sharpened FAIL→PASS exit criterion: injects corrupted D and proves the swap localizes correctly.
- Phase 1 `synthesize_tier_b2_backward → Err(NotImplemented)` stub removed; selector wrapper now routes through the real emitter.
- `crates/nsl-codegen/tests/tier_b2_ascii_only_ptx.rs` — ASCII-only invariant guardrail: every byte of emitted PTX must be 7-bit ASCII. Catches Unicode characters in `//` comments that cause cudarc's ptxas JIT to abort with `CUDA_ERROR_INVALID_PTX`. First-incident origin: 2026-05-20 D pre-pass launch failed because of em-dash + multiplication-sign characters in section comments.

### Validated on GPU (RTX 5070 Ti sm_120, 2026-05-20)

- D pre-pass GPU validation: **max_abs = 0.0** (bit-exact vs CPU reference) at all 4 tested configurations: canonical `(b=1, h=1, s=32, hd=32)` plus sweep cases `(1,1,64,32)` / `(1,2,96,64)` / `(2,1,128,128)`. Tolerances 5e-3 / 2e-2 / 4e-2 not even relevant — match is exact.
- `run_d_prepass_on_gpu` cudarc launcher wired (via `nsl_kernel_launch` + `nsl_test_cuda_*` primitives).
- `run_b1_forward_for_test` + `run_dq_kernel_on_gpu` remain `unimplemented!()`, **explicitly gated on Phase 2.5**. The dQ-kernel emitter is currently a structural scaffold (sections + register decls + MMA chain + labels — verified by ~20 ptxas/structural tests) but is **not data-mobile**: cp.async loads, HBM address derivation, dS SMEM scatter, col-major K re-stage scatter, tile_skip predicate computation, MMA fragment row/col setup, and loop back-edges all ship as PTX comments rather than emitted instructions. A launched dQ-kernel would read uninitialized SMEM. Phase 2.5 fills the data-mobility gap and is the gate to dQ GPU validation (Tests 2 + 3 in `tier_b2_dq_kernel_cpu_reference.rs`).

### Changed
- Moved the root-level research PDFs into `docs/research/` so research artifacts live with the rest of the repository's research material.
- Refreshed `README.md` to reflect the current documentation layout and the current local validation snapshot instead of stale passing-test counts.
- Refreshed `SPECIFICATION.md` to match the workspace version in `Cargo.toml` (`0.9.0`) and point readers at the current docs/research layout and validation status.

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
