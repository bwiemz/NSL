<!-- owner: @bwiemz -->

# Roadmap

NSL's development is organized into milestones (M9-M62) grouped into phases (1-13) shipping as versions (v0.1-v1.2). The roadmap is living -- phases beyond the next version boundary are subject to re-ordering.

**Current version:** `0.9.1` (2026-03-26)

## Phase table

| Phase | Version | Milestones | Status |
|---|---|---|---|
| 1 | v0.1 | M9-M13 (runtime, shape checking, model, autodiff, imports) | Shipped 2026-03-12 |
| 2 | v0.1 | M14 (training DSL), M17 (GPU/CUDA) | Shipped 2026-03-12 |
| 3 | v0.2 | M23-M31 (BYOD datatypes to graph fusion) | Shipped 2026-03-15 |
| 4 | v0.3 | M38a (linear types, semantics), M32 (MoE), M33 (spec dec), M34 (ring attn), M35 (FP8/AWQ/GPTQ), M36 (mem planning), M37 (roofline), M39a (vmap analysis), M40a (source AD analysis) | Shipped 2026-03-17 |
| 5 | v0.4-v0.5 | M41 (disagg inference), M42 (KV compress), M44 (constrained decode), M47 (multi-backend subset), M39b (vmap transform) | Shipped 2026-03-18 |
| 6 | v0.6 | M41 full (NVLink/RDMA/TCP KV backends), M47 full (KIR + backends), M39b AST transform | Shipped 2026-03-18 |
| 7 | v0.6-v0.7 | M38b (linear types codegen), M40b (source AD extraction), M43 (pipeline parallel) | Shipped 2026-03-18 |
| 8 | v0.8 | M45 (tensor debugger), M46 (reproducibility), M48 (multimodal) | Shipped 2026-03-18 |
| 9 | v0.8 | M49 (shape algebra), M50 (sparse tensors), M51 (effect system) | Shipped 2026-03-18 |
| 10 | v0.9 | M52 (weight-aware compilation), M62 (PyTorch FFI), M54 (unikernels) | In flight as of 2026-04-21 |
| 11 | v1.0 | M53 (WCET proofs), M55 (ZK circuits), M56 (agent shared memory) | Planned |
| 12 | v1.1 | M58 (elastic FT), M59 (topology routing), M61 (cluster debug) | Planned |
| 13 | v1.2 | M57 (FPGA/neuromorphic), M60 (exabyte streaming) | Planned |

> **Note on phases 4-9:** v0.3.0 through v0.8.0 shipped within a week (2026-03-17 to 2026-03-18) as a rapid catch-up release train. The phase boundaries above reflect the logical groupings from the roadmap design, not a multi-week development cycle.

## Milestone one-liners

### M9 -- Rust runtime + `nsl run` (Shipped v0.1, 2026-03-12)
Native Rust runtime library replacing the original C runtime (~770 LOC), plus the `nsl run` CLI entry point.

### M10 -- Compile-time tensor type system (Shipped v0.1, 2026-03-12)
`Tensor<[shape], dtype, device>` with named dimensions and compile-time shape verification at semantic analysis.

### M11 -- `model` keyword + model codegen (Shipped v0.1, 2026-03-12)
`model` blocks compiled to structs with forward/parameter methods; native `.nslm` binary serialization.

### M12 -- `grad` keyword + tape-based autodiff (Shipped v0.1, 2026-03-12)
First-class `grad` keyword triggering tape-based reverse-mode AD; `@no_grad`, `@checkpoint`, `@backward` decorators.

### M13 -- Import system + multi-file compilation (Shipped v0.1, 2026-03-12)
Module resolution, import graph construction, and incremental multi-file compilation.

### M14 -- Training DSL (Shipped v0.1, 2026-03-12)
`train` block lowered to epoch loops with implicit tape start/backward/stop; AdamW/Lion/Muon optimizers; schedulers auto-imported.

### M15-M16 -- Data pipeline + tokenization (Shipped v0.1, 2026-03-12)
`nsl.data` JSONL/CSV/mmap DataLoader; `nsl.tokenize` byte tokenizer and BPE encode/decode.

### M17 -- GPU/CUDA + `kernel` keyword (Shipped v0.1, 2026-03-12)
CUDA 13.x backend via cudarc 0.19; 15 PTX kernels; `kernel` blocks emit custom PTX; `.to(cuda)` device transfer.

### M18 -- Interop (Shipped v0.1, 2026-03-12)
Safetensors read/write; HuggingFace Hub model loading (single + sharded); ONNX export.

### M19-M20 -- Package ecosystem + v0.1 release (Shipped v0.1, 2026-03-12)
CLI commands (`nsl init`, `nsl fmt`, `nsl export`), project scaffolding, and the v0.1.0 public release.

### M21-M22 -- INT4/INT8 + FP8 quantization (Shipped v0.1-v0.2)
`quant` block with INT4/INT8 per-channel/group granularity; `nsl.quant` stdlib (quantize/dequantize).

### M23 -- BYOD custom datatypes (Shipped v0.2, 2026-03-15)
`datatype` block with `@pack`/`@unpack` for user-defined numeric formats; NslTensor.dtype widened to u16.

### M24 -- Zero-dependency standalone export (Shipped v0.2, 2026-03-15)
`nsl build --standalone` embeds weights into the binary; mmap sidecar backend; zero third-party runtime required.

### M25 -- PagedAttention + memory profiling (Shipped v0.2, 2026-03-15)
Paged KV-cache with BlockAllocator and PageTable; `@paged_kv` decorator; `--profile-memory` and Chrome tracing.

### M26 -- `@autotune` + elementwise fusion (Shipped v0.2, 2026-03-15)
Build-time Cartesian-product tuner; `@fuse` elementwise chain detection; fused PTX synthesis; `--profile-kernels`.

### M27 -- FlashAttention-2 (Shipped v0.2, 2026-03-15)
Tiled FA-2 PTX with 5 kernel variants; RoPE/GQA fusion; `@flash_attention`, `@rope`, `@gqa` decorators.

### M28 -- Dynamic shapes + ragged tensors (Shipped v0.2, 2026-03-15)
Symbolic dimension tracking; bounded syntax (`SeqLen < 4096`); runtime dimension assertions.

### M29 -- Continuous batching + serving engine (Shipped v0.2, 2026-03-15)
`serve` block frontend; chunked prefill with `RaggedBatchBuilder`; preemption manager (swap/recompute).

### M30 -- Tensor parallelism + NCCL (Shipped v0.2, 2026-03-15)
`@shard` decorator; SPMD process launcher; all-reduce/all-gather/broadcast FFI; weight shard copy.

### M31 -- Graph-level operator fusion (Shipped v0.2, 2026-03-15)
`FusionGraph` DAG; epilogue fusion (matmul+bias+activation); reduction fusion (softmax, layernorm, rmsnorm); `@fuse_graph`.

### M32 -- Mixture of Experts (Shipped v0.3, 2026-03-17)
`@moe` annotation; top-k gating; capacity routing; auxiliary load-balancing loss.

### M33 -- Speculative Decoding (Shipped v0.3, 2026-03-17)
`@speculative` decorator; tree attention; rejection sampling for speculative token acceptance.

### M34 -- Ring Attention (Shipped v0.3, 2026-03-17)
`@context_parallel` for cross-GPU sequence parallelism over arbitrarily long sequences.

### M35 -- FP8/AWQ/GPTQ quantization (Shipped v0.3, 2026-03-17; extended v0.9.1)
FP8 MMA paths; AWQ calibration pipeline; GPTQ full OBQ algorithm with Hessian-based error compensation (v0.9.1). Design: [`2026-04-15-m35-fp8-mma-correctness-design.md`](../superpowers/specs/2026-04-15-m35-fp8-mma-correctness-design.md).

### M36 -- Memory planning (Shipped v0.3, 2026-03-17)
Compile-time tensor liveness analysis; slab allocator for static memory budgets.

### M37 -- Roofline cost model (Shipped v0.3, 2026-03-17)
Per-op FLOP/byte analysis; roofline chart output; bandwidth-bound vs compute-bound classification.

### M38a -- Linear types semantics (Shipped v0.3, 2026-03-17)
Ownership checker behind a feature flag; `@shared` annotation; stdlib rewritten to be ownership-correct from day one.

### M38b -- Linear types codegen (Shipped v0.7, 2026-03-18)
Ownership decisions lowered to codegen; tensor lifetime proofs emitted; safety rules stabilized over v0.3-v0.6.

### M39 -- vmap (Shipped v0.3 analysis / v0.6 transform, 2026-03-17-18)
Batch dimension tracking and shape rewriting (M39a); `VmapTransformer` FnDef->FnDef AST rewriting (M39b).

### M40 -- Source-to-Source AD (Shipped v0.3 analysis / v0.7 extraction, 2026-03-17-18)
Wengert list construction and adjoint rules (M40a); source extraction and backward context wiring (M40b).

### M41 -- Disaggregated inference (Shipped v0.4-v0.6, 2026-03-18)
Prefill/decode worker separation; KV transfer abstraction; NVLink, RDMA, and TCP backends (M41b, v0.9.1).

### M42 -- KV-cache compression (Shipped v0.5, 2026-03-18)
INT8/INT4/FP8 KV compression; sliding-window eviction; H2O token-importance eviction policy.

### M43 -- Pipeline parallelism (Shipped v0.7, 2026-03-18)
1F1B and GPipe micro-batch scheduling; 3D rank mapping (tensor x pipeline x data); ZeRO-style gradient sharding.

### M44 -- Constrained decoding (Shipped v0.5, 2026-03-18)
Compiled FSM from grammar; token-level DFA for logit masking; zero-overhead at inference time.

### M45 -- Tensor debugger (Shipped v0.8, 2026-03-18)
Trace recording with NaN analysis; trace diffing across runs; Chrome tracing JSON export.

### M46 -- Reproducibility (Shipped v0.8, 2026-03-18)
Determinism checker; kernel variant selection for bitwise reproducibility; RNG state tracking.

### M47 -- Multi-backend KIR (Shipped v0.6, 2026-03-18)
Kernel IR (KIR) intermediate representation; PTX backend; `GpuTarget`/`GpuBackend` trait -- dense inference only (FlashAttention/MoE stay CUDA-only).

### M48 -- Multimodal (Shipped v0.8, 2026-03-18)
`PatchEmbed` (vision), `MelSpectrogram` (audio), `cross_attention` (cross-modal fusion), modality classification.

### M49 -- Shape algebra (Shipped v0.8, 2026-03-18)
Symbolic dimension solver; equality, divisibility, and range proofs at compile time.

### M50 -- Sparse tensors (Shipped v0.8, 2026-03-18)
`NslSparseTensor`; COO/CSR/CSC/BSR format dispatch; sparse-dense matmul routing.

### M51 -- Effect system (Shipped v0.8, 2026-03-18)
Effect annotations on functions; pure/stateful/io effect tracking; required by M46 determinism proofs and M49 shape algebra.

### M52 -- Weight-aware compilation (In flight, Phase 10)
Compiles model weights into the binary as compile-time constants; constant folding over weight tensors; CPDT Phase 1 shipped 2026-04-21, Phase 2 measurement-triggered. Design: [`2026-03-19-m52-m62-roadmap-design.md`](../plans/2026-03-19-m52-m62-roadmap-design.md).

### M53 -- WCET proofs (Planned, Phase 11)
Worst-case execution time analysis for robotics/safety-critical deployments; requires M52 weight constants. Design: [`2026-03-19-m52-m62-roadmap-design.md`](../plans/2026-03-19-m52-m62-roadmap-design.md).

### M54 -- Bare-metal unikernels (In flight, Phase 10)
x86_64 boot stub + unikernel runtime + GPU init (M54b shipped v0.9.1); single-binary AI inference with no OS. Design: [`2026-03-19-m52-m62-roadmap-design.md`](../plans/2026-03-19-m52-m62-roadmap-design.md).

### M55 -- ZK inference circuits (v1 shipped: folding backend)
Zero-knowledge proofs over model inference; verifiable computation for privacy-sensitive deployments. The folding backend is shipped and wired end-to-end (`nsl build --zk-backend folding` proves + `nsl zk verify`); the `halo2` and `plonky3` backends are refused at compile time (halo2 deprecated/removed, plonky3 prover not yet wired) rather than silently falling back to folding. Design: [`2026-03-19-m52-m62-roadmap-design.md`](../plans/2026-03-19-m52-m62-roadmap-design.md).

### M56 -- Multi-agent shared memory (Planned, Phase 11)
Safe shared-memory protocol for multi-agent systems; ownership enforced by M38 linear types. Design: [`2026-03-19-m56-multi-agent-design.md`](../superpowers/specs/2026-03-19-m56-multi-agent-design.md).

### M57 -- FPGA/neuromorphic backend (v1 shipped: clocked-FSM MLP; parity gated on external tools)
Emit to FPGA HLS or neuromorphic hardware targets via the M47 KIR; dense inference only. The v1 sequential clocked-FSM emitter for MLPs is shipped (PR #211); Verilator/Yosys synthesis-parity validation is gated on those external tools (not available in the default CI/dev environment). Further scope is paused pending re-scoping. Design: [`2026-03-19-m57-fpga-neuromorphic-design.md`](../superpowers/specs/2026-03-19-m57-fpga-neuromorphic-design.md).

### M58 -- Elastic fault tolerance (Planned, Phase 12)
Checkpoint/restore with worker elasticity; re-route around failed nodes mid-training. Design: [`2026-03-19-m58-fault-tolerance-design.md`](../superpowers/specs/2026-03-19-m58-fault-tolerance-design.md).

### M59 -- Topology-aware routing (Planned, Phase 12)
NVLink/IB topology discovery; placement optimization for multi-GPU communication patterns. Design: [`2026-03-19-m59-topology-routing-design.md`](../superpowers/specs/2026-03-19-m59-topology-routing-design.md).

### M60 -- Exabyte data streaming (Planned, Phase 13)
Distributed data pipeline for exabyte-scale corpora; streaming without full dataset materialization. Design: [`2026-03-19-m60-distributed-data-design.md`](../superpowers/specs/2026-03-19-m60-distributed-data-design.md).

### M61 -- Cluster debugging (Planned, Phase 12)
Cross-node trace correlation; per-rank NaN/divergence detection; cluster-level Chrome trace export. Design: [`2026-03-19-m61-cluster-debugging-design.md`](../superpowers/specs/2026-03-19-m61-cluster-debugging-design.md).

### M62 -- Legacy interop / PyTorch FFI (In flight, Phase 10)
`from_torch()`/`to_torch()` round-trips; `@export` decorator shipped 2026-04-15; grad-context bridge shipped 2026-04-16; per-function C wrappers and Python E2E tests pending. Design: [`2026-03-19-m62-legacy-interop-design.md`](../superpowers/specs/2026-03-19-m62-legacy-interop-design.md).

## Currently in flight

Cross-verified against `git log` as of commit `9a1b512e` (2026-04-21):

- **M52 CPDT Phase 2** -- weight-aware spectral factor; fires when committed-fixture disagreement exceeds 20% (currently 15.6%); measurement-triggered, not yet scheduled. Design: `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md`.
- **M62 PyTorch FFI -- per-function C wrappers** -- `@export` decorator and grad-context bridge shipped; per-function C wrappers and Python E2E tests remain. Design: `docs/superpowers/specs/2026-04-15-m62-c-wrappers-design.md`.
- **AWQ retention -- subprocess gaps** -- *shipped*. Original arena-ordering fix in PR #98; subprocess model-forward linkage and the model-library calibration link landed across PRs #134 + #145. The `awq_full_pipeline::end_to_end_real_subprocess_matches_analytical_reference` and `snapshot_awq_sidecar_baseline` tests both pass on `main`. No longer blocking WGGO Phase 2.
- **WGGO Phase 2 gradient scoring** -- *shipped*. Trait, scorers, sidecar schema, and hook were always in place; real backward-pass execution unblocked by the AWQ retention subprocess fix above and a six-hop calibration runtime chain (PRs #144 / #146 / #148 / #149). The merge-gate test `wggo_backward_pipeline::end_to_end_backward_subprocess_matches_analytical_reference` runs the full calibration subprocess, reads back populated `wggo_head_gradients` from the sidecar, and matches the analytical reference within 5×10⁻³ relative tolerance. `CalibratedGradientScorer` now consumes those real per-head scores via the `Task 30` integration tests.
- **CSHA Tier B** -- Section 9.2 Level 2 pipelining PTX (double-buffered async loads); not yet started; Tier A and C both shipped.
- **CSHA Tier D** -- Section 3.3 per-head mixed precision; not yet started; planned after Tier B.
- **WRGA B.3.2 fused backward** -- *deferred indefinitely (resolved 2026-05-23)*. The per-op profiling bench (`wrga_b32_per_op_breakdown.rs`) ran on current `main`: wall 6.6s/iter, GPU 86.3% of wall (NOT host/allocator-dominated), and **backward is 0.37x the forward, not >2.5x** -- the STUB's trigger condition fails on a clean baseline. The original 106x trigger was measured on a broken substrate (pre-cuBLAS / CPU-fallback / zero-duration profiler). A fused backward kernel would target at most 15% of GPU time. Decision tree resolved in `docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md`.
- **WRGA B.4 fused-forward staging** -- *resolved 2026-05-23: NOT pursued; fused stays opt-in, unfused cuBLAS is the default everywhere*. The fused GatedLoRA forward kernel is 73.4% of GPU time, but an m-sweep shows it **loses to cuBLAS at every m** (15x slower at m=1, widening to 68x at m=1024) -- there is no regime where it wins, so there is no niche to optimize the staging for. The assumed small-m niche (fusion's launch/round-trip savings dominate) was falsified: the lane-0 staging tax scales with n-block count (stays large at small-m), so the kernel's own cost dwarfs the savings ~10x. The staging rewrite is a speculative bet, gated on a concrete deployment need + a re-measured prototype that beats unfused. Beating cuBLAS at scale would need a multi-warp + tensor-core rewrite (a different, larger project). A separate robustness bug -- forward-only `@adapter` on sm>=80 segfaults because the side-table is materialized train-block-only -- is fixed independently. Full record: `docs/plans/2026-05-23-wrga-b4-fused-forward-staging-scope.md`.

Items removed from in-flight because git shows shipped: CSHA Gap J NaN (PR #103), CSHA block_q asymmetry (PR #101), CPDT Phase 1 + Tier A follow-ups (PRs #88-#94), PCA Tier A (PRs #78+#105+#109), WRGA B.3.2 Option 3 (PR #93), CSHA Tier A e2e (2026-04-15), CSHA Tier C close-out (2026-04-16), CSHA dead-head elim backward (PR #110), MSE backward /N fix (commit `c215194b`).

## How to pick up a milestone

1. **Check the milestone spec** in `docs/superpowers/specs/` (local, gitignored) or `docs/plans/`. Milestones progress: spec -> plan -> branch -> PR.
2. **Check for an open branch** -- `git branch -a | grep -i <milestone>` often shows partial work.
3. **Open an issue** on GitHub announcing intent to pick up -- this is how you claim a milestone.
4. **Branch naming** -- `feat/m<N>-<short-slug>` (e.g., `feat/m45-tensor-debugger`).
5. **Definition of done:**
   - Spec implemented
   - Unit + snapshot tests added
   - E2E example if user-facing
   - `cargo clippy -- -D warnings` clean
   - Design spec updated if implementation diverged

## How this roadmap changes

The M32-M51 phase ordering was revised once (2026-03-15 design doc). Phase 4 re-ordered after user feedback on inference-first identity. This is the expected behavior of a roadmap for a pre-1.0 compiler -- do not treat it as a schedule, treat it as a topological ordering that relaxes at phase boundaries.

What is load-bearing and won't move:

- **M38a (linear types semantics) ships before any stdlib rewrite** because the stdlib must be ownership-correct from day one.
- **CUDA-first** -- M47 multi-backend only covers dense inference; FlashAttention / Ring Attention / FP8 / MoE stay CUDA-only until there is demonstrated demand.
- **Inference-first identity** -- debugging, determinism, constrained decoding, and KV compression land before training niceties like source AD and pipeline parallelism.

## References

- Phase 10-13 (M52-M62): [`docs/plans/2026-03-19-m52-m62-roadmap-design.md`](../plans/2026-03-19-m52-m62-roadmap-design.md)

> **Note:** The phase 1-9 roadmap design files (`2026-03-09-nsl-roadmap-design.md`, `2026-03-13-m23-m31-roadmap-design.md`, `2026-03-15-m32-m51-roadmap-design.md`) are not present in this worktree's `docs/plans/` directory. The CHANGELOG and MEMORY.md are the authoritative records for phases 1-9 status.

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21.*
