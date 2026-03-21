# NeuralScript Development Roadmap: M52-M62

**Date:** 2026-03-19 (revised 2026-03-20)
**Status:** Active — M52, M53, M54, M55, M62 partially implemented (v0.9.0)
**Prerequisite:** v0.8.0 (M32-M51 complete, all b-milestones shipped)
**Architecture:** Rust compiler + Rust runtime + NSL stdlib + Cranelift AOT + PTX GPU (no C/C++/Python)
**Research validation:** 2026-03-20 — cross-referenced against NotebookLM research notebooks (competitive landscape, frontier features, performance theory)

---

## Vision

M32-M51 made NeuralScript a complete compiler with inference scaling, quantization, compile-time safety, and multi-backend support. M52-M62 transforms NSL from a *better ML framework* into a **new category of tool** — features that are not just faster than Python but *structurally impossible* in any interpreted, dynamically-typed, or JIT-compiled system.

These milestones target three frontiers:

1. **Weight-aware compilation**: The compiler consumes the actual model weights at build time, enabling constant folding, dead weight elimination, and bespoke binary generation per checkpoint.
2. **Safety-critical & verifiable AI**: WCET proofs for robotics, ZK proofs for verifiable inference, and multi-agent memory safety — markets where Python is literally banned.
3. **Frontier scale (100k+ GPUs)**: Elastic fault tolerance, topology-aware routing, and exabyte data streaming — features needed by the labs training the next generation of models.

### Strategic Principles

1. **M52 (Weight-Aware Compilation) is the flagship.** This is the single feature most likely to make NSL famous. No other system can compile model weights into the binary. It must be in Phase 10.
2. **M62 (Legacy Interop) is the adoption gate.** No frontier lab will adopt NSL without PyTorch FFI. This must also be Phase 10 — it unblocks adoption of everything else.
3. **Safety features (M53, M55) open new markets.** Robotics and verifiable AI are markets where Python cannot compete at all, not just where it's slower.
4. **Frontier scale (M58-M60) is where the money is.** But these features only matter once labs are actually using NSL, which requires M52 + M62 first.

---

## Dependency Graph

```
Phase 10: Weight Intelligence & Adoption Gateway (v0.9.0)
═════════════════════════════════════════════════════════
M52 (Weight-Aware Compilation) ←── M36 (Memory Planning), M24 (Standalone Export)
M62 (Legacy Interop / PyTorch FFI) ←── M18 (Interop, done)
M54 (Bare-Metal Unikernels) ←── M24 (Standalone), M29 (Serve)
M52 ──> M53, M55 (weight-aware IR enables WCET proofs and ZK circuits)
M62 ──> adoption of all subsequent milestones

Phase 11: Safety, Verification & Multi-Agent (v1.0.0)
══════════════════════════════════════════════════════
M53 (WCET Proofs) ←── M36 (Memory Planning), M37 (Roofline), M52 (Weight-Aware)
M55 (ZK Inference Circuits) ←── M52 (Weight-Aware, static DAG)
M56 (Multi-Agent Shared Memory) ←── M38 (Linear Types), M51 (Effect System)

Phase 12: Frontier Scale (v1.1.0)
═════════════════════════════════
M58 (Elastic Fault Tolerance) ←── M30 (Tensor Parallel), M43 (Pipeline Parallel)
M59 (Topology-Aware Routing) ←── M30 (NCCL), M34 (Ring Attention)
M61 (Cluster Time-Travel Debugging) ←── M45 (Tensor Debugger), M58 (Elastic)

Phase 13: Next-Gen Hardware & Data (v1.2.0)
═══════════════════════════════════════════
M57 (FPGA/Neuromorphic Backend) ←── M47 (KIR Multi-Backend)
M60 (Exabyte Distributed Data) ←── M19 (Data Pipeline, done)
```

**Critical paths:**
1. **M52 → M53 + M55**: Weight-aware static DAG enables both WCET proofs and ZK circuit emission.
2. **M62 → adoption → M58-M60**: Labs adopt NSL via PyTorch FFI, then need frontier scale.
3. **M38 + M51 → M56**: Linear types + effect system prove multi-agent memory safety.
4. **M47 KIR → M57**: Kernel IR abstraction extends to non-GPU hardware.
5. **M58 → M61**: Elastic execution is prerequisite for cluster-wide debugging.

---

## Phase 10: Weight Intelligence & Adoption Gateway (v0.9.0)

### M52: Weight-Aware Compilation & Network Constant Folding

**Goal:** Feed `.safetensors` weights to the compiler at build time. The compiler treats weights as `const` arrays, enabling dead weight elimination, sparsity-aware code generation, and per-checkpoint bespoke binaries.

**Why this changes everything:** Today, compilers treat architecture (math) and weights (data) separately. A generic `matmul` kernel processes any weight matrix identically. But if the compiler *knows* that 40% of a weight matrix is near-zero, it can physically delete those multiply-accumulate instructions. The resulting binary is not "a model runner" — it is "a binary that computes *this specific model's* output."

**Key components:**

- `nsl build --weights model.safetensors` CLI flag — weights loaded at compile time
- Weight constant folding: propagate known weight values through the computation graph (analogous to XLA's `HloConstantFolding`, TVM's `FoldConstant`, MLIR's `-sccp` pass — but NSL folds through the full tensor graph, not just scalar constants)
- Dead weight elimination: prune near-zero weights and their associated compute
- Sparsity-aware codegen: emit specialized kernels for weight matrices with known sparsity patterns, including NVIDIA structured 2:4 sparsity on Ampere/Hopper (MLIR's `gpu.create_2to4_spmat` proves this is viable at the IR level)
- Scaling constant fusion: fold quantization scales directly into PTX instructions
- Per-checkpoint binary: each build produces a binary optimized for that exact set of weights
- Compile-time model surgery: optimizer can prove which layers are identity-like and eliminate them
- Weight format support: `.safetensors` (primary, zero-copy mmap), `.gguf` (llama.cpp ecosystem — widely used for quantized models in the open-source community)

**Note on uniqueness:** Constant folding with known weights is standard (XLA, TVM, MLIR, ONNX Runtime all do it). NSL's unique contribution is per-checkpoint bespoke binary generation — the final binary is not "a model runner" but "a binary that computes this specific model's output," with dead weights physically absent from the executable.

**Codegen changes:** New `WeightAwarePass` between semantic analysis and Cranelift emission. Loads safetensors/GGUF into a `WeightMap`, annotates tensor operations with known values, runs constant propagation and dead code elimination on the tensor graph.

**Spec:** `docs/superpowers/specs/2026-03-19-m52-weight-aware-compilation-design.md`

---

### M62: Legacy Interop — Zero-Copy PyTorch/JAX FFI

**Goal:** Allow researchers to call NSL-compiled models from Python, and call Python functions from NSL, with zero-copy tensor sharing. This is the adoption gate — labs adopt NSL one layer at a time.

**Why this is critical:** No frontier lab will rewrite their entire data pipeline, evaluation harness, and experiment tracking in a new language. NSL must interoperate seamlessly with existing Python code.

**Key components:**
- `nsl build --shared-lib` — compile NSL model to a `.so`/`.dylib`/`.dll` with C API
- Python bindings (`pip install nslpy`) wrapping the C API with numpy/torch tensor interop
- Zero-copy tensor bridge: share memory between NslTensor and torch.Tensor/np.ndarray via DLPack
- Bidirectional calls: NSL calls Python (via `py.call()` existing M18 support), Python calls NSL
- Gradient bridge: NSL backward pass produces gradients compatible with torch.autograd
- HuggingFace pipeline integration: `nslpy.from_pretrained("llama-3")` loads weights into NSL binary
- ONNX Runtime custom op: register NSL-compiled subgraphs as ONNX Runtime ops

**Runtime additions:** `nsl_interop_dlpack_to_tensor`, `nsl_interop_tensor_to_dlpack`, `nsl_model_forward_c_api`

**Spec:** `docs/superpowers/specs/2026-03-19-m62-legacy-interop-design.md`

---

### M54: Bare-Metal Unikernels — Zero-OS Deployment

**Goal:** Compile the entire inference stack (model + HTTP server + KV-cache) into a bootable hypervisor image that runs without an operating system. 50MB image, millisecond boot, NIC-to-GPU direct path.

**Why this matters:** Deploying a PyTorch model requires a 2GB Docker container with Ubuntu, glibc, Python, and the NVIDIA driver stack. A unikernel eliminates the OS entirely — the model *is* the operating system.

**Key components:**
- `nsl build --unikernel` — emits a bootable x86_64 ELF with embedded model + serve stack
- Minimal runtime: virtio-net driver, PCI-e enumeration for GPU, memory-mapped I/O
- GPU initialization without OS: direct CUDA Driver API calls from ring-0
- HTTP server compiled into the binary (no nginx/gunicorn/uvicorn)
- Hypervisor targets: KVM/QEMU, AWS Firecracker, Google gVisor
- Boot sequence: UEFI → GPU init → model weight load → HTTP listen (< 500ms)
- Memory: entire DRAM is the model's memory pool — no OS overhead, no page table walks

**Codegen changes:** New `UnikernelLinker` that produces a standalone ELF with custom entry point, interrupt handlers, and embedded PCI-e/virtio drivers (using Rust `no_std` crates).

**Spec:** `docs/superpowers/specs/2026-03-19-m54-unikernel-design.md`

---

## Phase 11: Safety, Verification & Multi-Agent (v1.0.0)

*These features open markets where Python is literally banned.*

### M53: Hard Real-Time AI — WCET Proofs for Robotics/Aerospace

**Goal:** The compiler generates a mathematical proof that a forward pass will complete within a bounded time, with zero variance. NSL becomes the default language for safety-critical edge AI.

**Why this is unique to NSL:** Python's garbage collection and dynamic dispatch cause unpredictable latency spikes. NSL has static memory planning (M36, zero dynamic allocation), a roofline cost model (M37, known instruction counts), and weight-aware compilation (M52, known data). These three together enable precise WCET calculation.

**CRITICAL RESEARCH FINDING (2026-03-20):** GPU WCET is considered **intractable** by the static analysis community. NVIDIA GPUs have undocumented, proprietary warp-scheduling logic and DDR scheduling queues. Static tools (aiT, Chronos, SWEET) only work on CPUs with known cache replacement policies (e.g., LRU) and bounded pipelines. Floating-point non-associativity (FPNA) combined with asynchronous parallel reductions causes unpredictable execution traces. **NSL cannot offer hard WCET guarantees on NVIDIA GPUs.**

**Revised strategy:** Two-tier WCET:

1. **Tier 1 — FPGA path (hard real-time):** Emit Xilinx XIR/DPU instructions for FPGA targets where timing is deterministic. Use On-Chip Memory (OCM) for instruction fetch (DICTAT-style isolation) to eliminate AXI bus contention. This is the only path that can produce certified WCET bounds (DO-178C, ISO 26262).
2. **Tier 2 — GPU path (soft real-time):** Provide statistical latency bounds based on roofline cost model + empirical profiling. Useful for latency SLOs but NOT for safety certification. The Groq LPU is deterministic by default and could serve as an alternative hard-RT GPU-class target.

**Key components:**

- `nsl build --wcet --target fpga` — hard real-time FPGA path (certified)
- `nsl build --wcet --target gpu` — soft real-time GPU path (statistical bounds)
- Per-operation cycle counting: each tensor op maps to exact cycle count on FPGA, estimated on GPU
- Memory access pattern analysis: prove no cache misses on FPGA (impossible to prove on GPU)
- `@real_time(max_latency_ms=15.0)` decorator — compile error if WCET exceeds bound (FPGA only)
- WCET certificate: JSON file with per-operation timing breakdown and proof chain
- DO-178C / ISO 26262 compliance: documentation generation for aerospace/automotive certification (FPGA path only)
- No heap allocation in certified path: compiler proves all memory is stack/slab-allocated
- Groq LPU backend: deterministic by hardware design — can provide hard WCET without FPGA (future)

**Semantic changes:** New `WCETAnalyzer` pass that walks the static computation graph, summing worst-case cycle counts. Uses FPGA timing models (deterministic) or GPU roofline estimates (statistical). Integrates with M37's hardware spec database.

**Spec:** `docs/superpowers/specs/2026-03-19-m53-wcet-proofs-design.md`

---

### M55: Zero-Knowledge Inference Circuits

**Goal:** Compile a model forward pass to a ZK-SNARK/STARK circuit, producing a mathematical proof that a specific output was generated by a specific model without revealing the weights or input.

**Why this is unique to NSL:** Generating ZK circuits from Python loops is excruciatingly slow because the circuit must capture every dynamic branch. NSL's static, shape-checked expression DAG (especially with M52's weight-aware constant folding) is the perfect IR for circuit emission — every operation is statically known.

**CRITICAL RESEARCH UPDATE (2026-03-20):** The ZK-ML landscape has evolved dramatically since the original design. EZKL/Halo2 scales poorly beyond ~30M parameters (148GB RAM for GPT-2, hours of proving). The state-of-the-art has shifted to:

- **ZKTorch**: Folding accumulation schemes, proves 6B-parameter GPT-J in ~20 minutes on 64 threads
- **zkLLM**: Parallelized tlookup argument, proves 13B-parameter LLaMA-2 with <200KB proof size
- **Jolt Atlas**: Lookup-native architecture (abandons PLONKish quotient polynomials entirely), 4-7x faster than CPU-based proofs
- **Lagrange DeepProve**: Sumcheck + logup GKR, 54-158x faster than EZKL for GPT-2
- **Field transition**: BN254 (254-bit) → Mersenne-31 (31-bit) yields ~10x faster field operations

**Revised approach:** Target lookup-native arithmetizations (Jolt-style) rather than generic PLONKish/Halo2. R1CS format is outdated — use AIR (Algebraic Intermediate Representation) or lookup-native IR.

**Key components:**

- `@zk_proof` decorator on model forward functions
- Expression DAG → lookup-native arithmetic circuit compiler (AIR format, not R1CS)
- Primary backend: Jolt Atlas-style lookup tables for non-linear activations (proven superior to polynomial approximation and bit-decomposition)
- Secondary backend: Halo2 (maintained for compatibility but not recommended for >30M params)
- Folding accumulation: compress proofs of deep networks via recursive folding (ZKTorch approach) — proof size decoupled from model depth
- Field target: Mersenne-31 (M31) for ~10x faster field ops vs. BN254 (with BN254 fallback for EVM compatibility)
- Quantized-friendly: INT8 models map naturally to finite field arithmetic
- Split proof: prove "output was generated by *a* model with *this* architecture" without revealing weights
- Verification contract: emit Solidity/Move smart contract for on-chain verification
- `nsl build --zk-circuit model.nsl` — produces circuit + verification key
- Target: 7B+ parameter models provable in minutes, not hours; proof sizes <500KB

**Codegen changes:** New `ZKCircuitEmitter` that translates the Wengert list (M40) into lookup-native arithmetic constraints. Fixed-point arithmetic for field-compatible computation. Arena-based memory allocation for cryptographic trace buffers (avoids OOM that plagues Python/Halo2 wrappers).

**Spec:** `docs/superpowers/specs/2026-03-19-m55-zk-inference-design.md`

---

### M56: Natively Compiled Multi-Agent Shared Memory

**Goal:** Compile multiple AI agents into a single binary where they share KV-cache and activations via zero-copy RAM transfers instead of JSON serialization over HTTP.

**Why this is unique to NSL:** Linear types (M38) prove Agent A won't corrupt Agent B's memory. The effect system (M51) proves agents don't have unexpected side effects. No other language can give these guarantees at compile time for concurrent AI workloads.

**RESEARCH UPDATE (2026-03-20):** No compiled system currently provides formal memory safety proofs for concurrent AI agents. Ray (actor model) incurs massive single-node overhead from lock acquisition and serialization. The state-of-the-art is the **Lingua Franca reactor model** — static dependency graphs with strict global logical time ordering, eliminating dynamic lock acquisition entirely. LF achieves 11.6x higher throughput than Ray in multi-agent RL and 5.12x faster inference.

**Revised approach:** Implement a reactor-model scheduler (not actor-model) where agent communication patterns are static and resolved at compile time. This eliminates IPC overhead entirely.

**Key components:**

- `agent` keyword — like `model` but with its own isolated KV-cache and state
- **Reactor-model scheduler**: agent communication graph is static, resolved at compile time (no dynamic lock acquisition, no runtime synchronization messages)
- `agent.send(other_agent, kv_cache_ptr)` — zero-copy KV-cache transfer between agents via in-memory object store (HPRM-style zero-copy adaptive serialization)
- Compile-time ownership proof: linear types verify no shared mutable state between agents
- Effect isolation: effect system proves agent functions are `Communication`-only (no `Mutation` cross-boundary)
- Shared embedding table: multiple agents can borrow the same weight tensors (via `@shared` or `&Tensor` borrows)
- `@pipeline_agent(agents=[drafter, reviewer, editor])` — declarative multi-agent pipeline with static dependency graph
- Global logical time ordering via Runtime Infrastructure (RTI) — ensures deterministic agent execution order
- Benchmarks target: 1000x faster than LangChain/AutoGPT JSON serialization, 10x faster than Ray actors

**Semantic changes:** New `AgentChecker` that verifies ownership boundaries between agent scopes. Extends `OwnershipChecker` (M38a) with cross-agent transfer rules. Static dependency graph extraction enables the reactor scheduler to completely eliminate dynamic synchronization.

**Spec:** `docs/superpowers/specs/2026-03-19-m56-multi-agent-design.md`

---

## Phase 12: Frontier Scale (v1.1.0)

*Features needed for 100,000+ GPU training runs.*

### M58: Cluster-Scale Fault Tolerance & Elastic Execution

**Goal:** When a GPU node dies during a 100k-GPU training run, the system automatically detects the failure, re-shards weights and KV-cache to surviving nodes, loads from the last micro-checkpoint, and resumes — without human intervention.

**Key components:**

- Async heartbeat monitor: each rank sends periodic heartbeats via out-of-band channel
- Failure detection: configurable timeout (default 30s), exponential backoff for transient failures
- Elastic `world_size`: `train { distribute: "dp=elastic, tp=4, pp=4" }` — DP dimension grows/shrinks
- Automatic re-sharding: when a node dies, redistribute DP replicas across survivors
- **Hierarchical micro-checkpointing** (ByteCheckpoint/Gemini pattern): frequent micro-checkpoints to CPU DRAM or local NVMe (fast), infrequent flushes to S3/shared filesystem (durable). Checkpoint interval optimized based on cluster failure probability.
- Zero-copy checkpoint via mmap: NSL's static typing and no-GC architecture enables direct memory-mapped weight dumps to NVMe via GPUDirect Storage, bypassing Python's pickle serialization entirely
- Resume protocol: load last checkpoint, rebuild NCCL communicators with new world_size, continue
- `@fault_tolerant` decorator on train blocks — enables all resilience features
- LR auto-adjustment on resize: `lr * (new_dp / old_dp)` maintains effective batch size semantics

**Runtime additions:** `nsl_heartbeat_start`, `nsl_heartbeat_stop`, `nsl_elastic_resize`, `nsl_checkpoint_async`, `nsl_checkpoint_mmap`

**Spec:** `docs/superpowers/specs/2026-03-19-m58-fault-tolerance-design.md`

---

### M59: Topology-Aware Network Routing

**Goal:** Feed the physical datacenter network topology to the compiler so it can optimize collective communication (all-reduce, all-gather, send/recv) to minimize cross-rack traffic.

**RESEARCH NOTE (2026-03-20):** NCCL already performs automatic topology discovery and constructs optimal ring/tree communication graphs at initialization. NSL's value-add is compile-time optimization *beyond* what NCCL does: (1) static rank placement before NCCL init, (2) direct SHARP (in-switch reduction) PTX emission bypassing the NCCL runtime, (3) compile-time routing tables embedded in the binary.

**Key components:**

- `nsl build --topology cluster.json` — hardware topology specification
- Topology-aware rank mapping: place TP groups within the same switch, PP stages across racks (done BEFORE NCCL init, not after)
- Hierarchical all-reduce: intra-node NVLink → intra-rack InfiniBand → inter-rack optical
- **SHARP integration**: Emit `multimem.red.release.sys.global.add` PTX instructions directly for NVLink in-switch reductions, bypassing NCCL's generic AllReduce path. Automatically inject `fence.proxy.alias` memory barriers for unicast/multicast consistency.
- Ring attention path optimization: ring order follows physical network shortest path
- Bandwidth-aware pipeline scheduling: more micro-batches for slower inter-rack links
- Topology specification format: JSON describing switches, NICs, bandwidths, latencies
- Compile-time routing table: optimal send/recv paths pre-computed, embedded in binary
- **NVLink-SHARP determinism**: On Hopper/Blackwell with CUDA 12.8+, in-switch reductions are hardware-deterministic — NSL can guarantee bitwise reproducibility for distributed training

**Codegen changes:** New `TopologyRouter` that reads cluster.json and emits optimized NCCL communicator initialization with topology-aware rank placement. `SharpEmitter` generates direct PTX for in-fabric compute operations.

**Spec:** `docs/superpowers/specs/2026-03-19-m59-topology-routing-design.md`

---

### M61: Time-Travel Cluster Debugging

**Goal:** When loss spikes to NaN at iteration 450,000 on a 100k-GPU run, identify exactly which GPU, which layer, and which operation caused the overflow — across the entire cluster.

**Key components:**
- `nsl run --cluster-trace` — distributed tracing across all ranks
- Per-rank binary trace files (M45 format) with global clock synchronization
- `nsl debug --cluster trace_dir/` — merges traces from all ranks, timeline view
- Cross-rank anomaly detection: find the first rank where NaN/Inf appears
- Gradient magnitude tracking: per-rank, per-layer gradient norm history
- Communication audit: verify all-reduce results match across ranks (detect bit-flip corruptions)
- "Freeze the cluster": on anomaly detection, all ranks dump state simultaneously

**Runtime additions:** `nsl_cluster_trace_init`, `nsl_cluster_trace_sync_clock`, `nsl_cluster_trace_anomaly_check`

**Spec:** `docs/superpowers/specs/2026-03-19-m61-cluster-debugging-design.md`

---

## Phase 13: Next-Gen Hardware & Data (v1.2.0)

### M57: FPGA & Neuromorphic Backend

**Goal:** Extend KIR (Kernel IR from M47) to emit Verilog for FPGAs and specialized assembly for neuromorphic/LPU chips. NSL becomes the frontend for next-gen AI hardware startups.

**RESEARCH UPDATE (2026-03-20):** Direct HLS-to-Verilog is NOT the state-of-the-art for FPGA ML inference. Modern tools (Vitis AI) use pre-synthesized DPU (Deep-Learning Processing Unit) IP cores and compile to XIR instruction streams. Raw HLS defaults to sequential task execution and often performs worse than CPU. Neuromorphic: Loihi 2 now supports graded spikes (SDNN — Sigma-Delta Neural Networks) enabling standard DNN inference, not just SNNs. RISC-V AI accelerators are viable via Cranelift's native RISC-V backend.

**Revised approach:** Two-track FPGA strategy + expanded hardware targets.

**Key components:**

- **FPGA Track 1 — XIR/DPU path** (recommended): Emit Xilinx XIR instruction streams for the pre-synthesized DPU IP core. Standard INT8 quantized layers (conv, linear, attention) map directly to DPU instructions. This achieves 34x speedup over ARM CPUs with zero custom RTL.
- **FPGA Track 2 — HLS fallback**: For exotic operators unsupported by DPU (3D conv, custom activations), emit Vitis HLS C++ with explicit parallelization directives. Without directives, HLS defaults to sequential execution — NSL must inject pipeline/unroll pragmas.
- `nsl build --target fpga --device xilinx-u280` — FPGA synthesis flow (auto-selects DPU vs HLS per operator)
- Systolic array mapping: partition matmul tiles into FPGA fabric (HLS path only)
- Resource budgeting: compiler proves the design fits within FPGA LUT/BRAM limits
- Latency-deterministic execution: FPGA execution has zero variance (feeds into M53 WCET)
- **OCM instruction isolation**: Map DPU instruction fetches to On-Chip Memory (DICTAT-style) to eliminate AXI bus contention and tighten WCET bounds by ~25%
- **Groq LPU backend**: Deterministic by hardware design (no non-deterministic execution mode exists). KIR maps to Groq TSP assembly. NSL can omit software synchronization barriers that GPU paths require.
- **Neuromorphic backend (Loihi 2)**: Target Loihi 2's programmable microcode instruction set (ALU ops: ADD, MUL_SHR, JMP_C). Use SDNN (Sigma-Delta) encoding for standard DNN inference with event-driven sparse communication. Compile NSL agent channels to Lava-compatible async message passing.
- **RISC-V AI accelerators**: Cranelift natively supports RISC-V emission. Target Tenstorrent and RISC Zero zkVM (for ZK proofs, though specialized circuits are 65x faster than generic zkVM).

**Codegen changes:** New `backend_xir.rs` (FPGA DPU), `backend_hls.rs` (FPGA custom ops), `backend_groq.rs` (Groq LPU), `backend_loihi.rs` (Loihi 2 microcode) modules implementing KIR lowering to hardware-specific formats.

**Spec:** `docs/superpowers/specs/2026-03-19-m57-fpga-neuromorphic-design.md`

---

### M60: Exabyte-Scale Distributed Data Streaming

**Goal:** Stream trillions of tokens from globally distributed storage into GPUs without ever letting the GPUs starve. Kernel-bypass I/O via RDMA.

**RESEARCH NOTE (2026-03-20):** GPUDirect Storage (GDS) is production-ready and fully integrated into Nsight Systems profiling on Linux. GDS enables DMA between NVMe and GPU memory via cuFile APIs, bypassing CPU bounce buffers. On Hopper, GDS can pipe directly into the Tensor Memory Accelerator (TMA) for zero-copy NVMe-to-shared-memory transfer. Mosaic StreamingDataset (binary shards, deterministic streaming) is the SOTA pattern for distributed data loading.

**Key components:**

- `data { source: "s3://bucket/training-data", shards: 10000 }` — distributed data declaration
- **GPUDirect Storage (GDS)**: cuFile API for DMA NVMe-to-GPU, bypassing CPU entirely. NSL's strict aliasing rules enable static proof that async GDS writes are complete before tensor core reads (via `mbarrier` hardware barriers).
- **TMA integration (Hopper)**: Pipe GDS directly into Tensor Memory Accelerator for multi-dimensional array prefetch into shared memory. Perfect overlap of NVMe I/O and matrix math.
- Prefetch pipeline: CPU nodes decode/tokenize data N steps ahead of GPU consumption
- **Mosaic-style binary shards**: Deterministic, scalable streaming across distributed nodes via pre-chunked binary shard files. Shard assignment is deterministic per-rank — no global coordination needed.
- Distributed shuffle: globally shuffled epoch without materializing the full dataset
- Multi-modal streaming: interleaved text/image/video/audio streams with format-aware decoding
- Backpressure: if GPUs are slower than data pipeline, throttle without dropping data
- Checkpoint-aware: data pipeline position saved with model checkpoint for exact resumption

**Runtime additions:** `nsl_data_stream_init`, `nsl_data_stream_next_batch`, `nsl_data_stream_checkpoint`, `nsl_gds_read_async`

**Spec:** `docs/superpowers/specs/2026-03-19-m60-distributed-data-design.md`

---

## Release Plan

| Version | Milestones | Theme | Notes |
|---------|-----------|-------|-------|
| v0.9.0 | M52, M62, M54 | Weight Intelligence & Adoption Gateway | M52 is the flagship; M62 unblocks adoption |
| v1.0.0 | M53, M55, M56 | Safety, Verification & Multi-Agent | Opens robotics, ZK, and agentic AI markets |
| v1.1.0 | M58, M59, M61 | Frontier Scale | Required for 100k+ GPU training |
| v1.2.0 | M57, M60 | Next-Gen Hardware & Data | Future hardware + exabyte data |

---

## Parallelization Opportunities

- **Phase 10:** M52 and M62 are independent. M54 depends on M52's standalone binary work but can overlap.
- **Phase 11:** M53, M55, M56 are all independent (M53/M55 share M52 dependency but different outputs).
- **Phase 12:** M58 and M59 are independent. M61 benefits from M58 but can start concurrently.
- **Phase 13:** M57 and M60 are fully independent.

---

## NSL's Moat Features (M52-M62)

These features are **structurally impossible** in Python/PyTorch/JAX:

| Feature | Why impossible in Python | Research validation |
|---------|------------------------|---------------------|
| M52 Weight-aware compilation | Requires AOT static graph + weight constants at compile time | Constant folding is standard (XLA/TVM/MLIR); per-checkpoint binaries are unique to NSL |
| M53 WCET proofs | Requires zero dynamic allocation + known instruction counts | GPU WCET is intractable (undocumented warp scheduling); FPGA path required for certification |
| M54 Unikernels | Requires no interpreter, no OS, no garbage collector | Validated — no competing approach |
| M55 ZK circuits | Requires static expression DAG, not dynamic Python loops | Lookup-native (Jolt) >> PLONKish (Halo2); folding schemes enable 7B+ params in minutes |
| M56 Multi-agent safety | Requires linear types + effect system for memory safety proofs | Reactor model (Lingua Franca) 11.6x faster than Ray; static dependency graphs key |
| M57 FPGA/neuromorphic | Requires static IR that maps to hardware description | XIR/DPU path preferred over raw Verilog; Loihi 2 SDNN enables standard DNN inference |
| M58 Elastic execution | Requires compiled SPMD with reshardable static graph | Hierarchical micro-checkpointing (DRAM→NVMe→S3) is SOTA pattern |
| M59 Topology routing | Requires compile-time network topology awareness | NCCL auto-discovers topology; NSL adds SHARP PTX emission + compile-time rank placement |

---

## Spec Documents

| Milestone | Spec Path |
|-----------|-----------|
| M52 | `docs/superpowers/specs/2026-03-19-m52-weight-aware-compilation-design.md` |
| M53 | `docs/superpowers/specs/2026-03-19-m53-wcet-proofs-design.md` |
| M54 | `docs/superpowers/specs/2026-03-19-m54-unikernel-design.md` |
| M55 | `docs/superpowers/specs/2026-03-19-m55-zk-inference-design.md` |
| M56 | `docs/superpowers/specs/2026-03-19-m56-multi-agent-design.md` |
| M57 | `docs/superpowers/specs/2026-03-19-m57-fpga-neuromorphic-design.md` |
| M58 | `docs/superpowers/specs/2026-03-19-m58-fault-tolerance-design.md` |
| M59 | `docs/superpowers/specs/2026-03-19-m59-topology-routing-design.md` |
| M60 | `docs/superpowers/specs/2026-03-19-m60-distributed-data-design.md` |
| M61 | `docs/superpowers/specs/2026-03-19-m61-cluster-debugging-design.md` |
| M62 | `docs/superpowers/specs/2026-03-19-m62-legacy-interop-design.md` |
