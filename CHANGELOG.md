# Changelog

All notable changes to NeuralScript will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed — P4 item 16: dtype ABI migration (both tag collisions removed)

- **`DTYPE_I32 = 9`**: i32 token tensors carry their own canonical tag.
  Tag 4 means `DTYPE_INT8` alone, with its true 1-byte width in
  `dtype_element_size`. Every producer/consumer migrated in lockstep:
  DataLoader batches, the CPU tensor factory, the `*i32` runtime readers,
  the GPU i32-index kernel dispatch (embedding fwd/bwd, gather), the
  fused-CE label decoder, the elementwise promote paths, and the
  collective byte-width table (which gains `I32=4`, closing a latent
  1-byte mis-size for i32 tensors through collectives). No on-disk format
  stores numeric tags, so the migration is disk-safe.
- **The C API speaks the canonical tag space verbatim** (`NslTensorDesc.
  dtype`: 0=f64, 1=f32, 2=f16, 3=bf16, 4=int8, ..., 9=i32). The
  historical inverted 0=f32/1=f64 convention is GONE; `capi_dtype_to_nsl`
  / `nsl_dtype_to_capi` remain only as validating identity chokepoints
  that abort on unknown tags instead of the old silent fall-back-to-f64
  (which mislabeled the never-mapped C-API int64/uint8 slots). Updated in
  lockstep: calibration-wrapper codegen immediates, the ONNX-RT element
  map (int64/uint8 now refused — they never reached a real compute path),
  the dispatch element-size table, the generated C header, and the Python
  ctypes mirrors. **Breaking for external C/ctypes callers** that baked
  the old convention; the `dtype_abi_lock` golden test pins the new one.

### Added — P4 item 17: SR-BF16 authoritative weights (`--param-dtype bf16-sr`)

- Every STREAMED parameter's authoritative copy is a device-resident
  BF16 buffer — 2 bytes/param, **no FP32 master copy, no host mirror**.
  Rides the weight-stream residency schedule: upload = device-side
  bf16→f32 widen into a transient working view (no PCIe), evict = free
  (no writeback — the fused SR optimizer step IS the persistence),
  teardown re-materializes plain f32 tensors for model_save/eval.
  Un-streamed params (view-rooted/tied — the set ZeRO-3 keeps
  Replicated) keep f32 authority through the plain fused step.
- The update is a fused AdamW PTX kernel with the f32 kernel's exact
  rounding sequence plus a stochastic-rounding tail: `splitmix64(seed ^
  step·SALT, param_idx≪40 + elem)` → 16-bit dither added to the f32
  result bits before bf16 truncation. **Compiler-owned counters**: the
  stream is a pure function of (`--seed`, step, param, element) —
  deterministic across reruns, ranks, and launch order. Explicit edge
  policy: rounding-induced overflow saturates to ±max-normal; arithmetic
  Inf propagates; NaN forced to quiet NaN; underflow gradual.
- FP32 gradients/reductions everywhere (m/v/m_partial stay f32).
  Refusals: requires `--weight-stream` + the FASE-Deferred fused
  AdamW/Adam shape; refuses Muon, ZeRO, offload, reduced-precision
  moments, WGGO per-layer overrides, `--training-reference`.
- Gates: exhaustive-dither exact-unbiasedness unit tests; GPU rounding
  tail bit-identical to the CPU reference over 65k adversarial values;
  e2e training with bit-identical same-seed reruns, seed-sensitive
  streams, and per-step tracking of the f32 baseline.

### Added — P4 item 18 rung 2: compressed Muon state (`--muon-state-dtype bf16`)

- Ladder order: f32 (default) → **bf16 momentum + f32 working buffer**
  (this rung) → blockwise 8-bit → 4-bit structural (later rungs refuse
  loudly with the ladder order in the message).
- `bf16` halves Muon first-moment memory: m allocates BF16; each CSLA
  group update dequants to f32, runs the unchanged stdlib `muon_step`,
  and quant-stores back with counter-based SR (salted separately from
  the item-17 weight stream). SR — not RTE — because the momentum EMA's
  `(1−β)·g` increment routinely falls below a bf16 ulp. `v` stays f32.
  Muon-only; requires `--layerwise-accum`; refuses ZeRO and offload.

### Added — P3: ZeRO-3 tensor-granular parameter sharding (items 12-14)

- **`--zero-stage 3` is lowered** on the layerwise residency schedule
  (requires `--layerwise-accum --weight-stream --checkpoint-blocks
  --source-ad`; anything else refuses with the flag list). Each parameter
  tensor is OWNED by one rank (the stage-1/2 byte-balanced partition);
  at rest the owner keeps it device-resident and every other rank holds
  NOTHING — per-rank at-rest parameter memory is ~1/ws. This is the
  per-parameter FSDP granularity (owner = singleton group per tensor):
  it reuses the existing owner maps, composes with Muon (whole matrices
  stay whole), and needs no gather in the optimizer. Elementwise 1/ws
  sharding via `all_gather` is the documented follow-up.
- **Item 12 — `ParameterResidency`** (`Replicated | ShardedResident |
  GatheredTemporary | Evicted`) tracked per replica in the runtime.
  Callbacks that touch θ ride the existing residency bracket
  (`upload_all`/`reevict_all` redirect to gather/release), `model_save`
  reads full replicas via the teardown restore, and tied/view-rooted
  params stay `Replicated` through the same view-rooted exclusion the
  weight streamer uses (updated identically on every rank from
  all-reduced gradients).
- **Item 13 — JIT gather per layer**: the weight-stream upload/evict
  sites (per-segment forward brackets, window range heads, packs,
  prefetch, async evict) redirect to a collective broadcast-fill /
  free-release backend when zero3 is active — GPU-only, no host mirrors
  involved. Registration evicts non-owner replicas on the first window.
- **Item 14 — comm ordering from source-AD readiness**: each layer
  group's gradient slots all-reduce (sum ÷ ws, the stage-1/2 averaging
  convention) at its group update — the exact point the layerwise
  schedule knows that layer's backward completed — then the owner
  updates and non-owners release; the next range head (or the prefetch
  edge, when `--stream-prefetch` is on) gathers the following layer.
  Under the sim/sim-gpu backends collectives are synchronous, so the
  ordering is validated bit-exactly; true comm/compute overlap on a
  dedicated stream is NCCL-gated follow-up (not reachable on a 1-GPU
  box).
- Constraints (loud refusals): stage 3 × `--optim-state-offload`,
  stage 3 × reduced-precision moments, stage ≥ 4. Optimizer state stays
  REPLICATED in v1 (the enable note says so).
- Gates (`zero3_gate.rs`): 2-rank sim-gpu training is BIT-IDENTICAL to
  the single-rank baseline (rank-0 loss stream + saved model bytes),
  with gather/release counters asserted non-vacuous; the same parity
  holds for Muon × zero3 × arena/prefetch/async-writeback × a callback
  that reads a sharded param mid-training; refusal coverage for the
  unsupported combos.

### Changed — P1: Muon validation + performance (items 5-11)

- **Parameter-ROLE routing replaces the name-substring exclusion list**
  (item 6). Mixed Muon/AdamW routing is now decided per parameter:
  explicit `@param_role("embedding"|"head"|"hidden")` field decorator
  (invalid values are compile errors) > structural inference (the table
  argument of `embedding_lookup(self.field, ids)` is role `embedding`;
  weight-tied heads are the same tensor) > declared rank != 2 (`vector`)
  > default `hidden`. A hidden weight named `embed_proj` now correctly
  takes Muon; an embedding named `tok_table` now correctly takes AdamW —
  both silently misrouted before. The routing table prints every param's
  role + provenance; untied-head models get a loud annotate-me note.
- **`adamw_lr` knob on `Muon(...)`** (item 5 prerequisite): the AdamW arm
  (embeddings/head/vectors) gets its own learning rate, threaded as a
  fixed ratio of `lr` so schedulers modulate both arms proportionally.
  Unset, it follows `lr` exactly (bit-exact with the old single-lr step).
- **`ns_steps` is validated** (item 7): floats, zero, and negatives are
  compile errors instead of silently degrading the NS iteration.
- **`.item()` removed from the Muon step / planned NS primitive**
  (items 8+10): `muon_step` now calls `muon_orthogonalize_fast`, a
  runtime primitive (`nsl_tensor_muon_orthogonalize`) that runs the
  quintic Newton-Schulz chain from Rust — on GPU the Frobenius
  pre-normalization is computed and consumed entirely on-device (stats
  kernel into a persistent 16-byte scratch + a scale kernel that reads
  it; no DtoH sync per rank-2 param per step), tall/wide handled by
  materialized transposes at entry/exit, intermediates recycled through
  the caching allocator. The NSL-level `muon_orthogonalize` stays in
  `nsl.optim.muon` as the pinned reference (CPU f64 gate: sum-sq diff
  < 1e-18; GPU gate bounds the f32 norm-path drift). Deferred:
  batched small-matrix NS, stochastic-rounded direct updates.
- **v (AdamW second moment) allocates only where the AdamW arm reads it**
  (item 9): Muon-routed params carry a null slot (exactly the runtime
  condition `muon_step` routes on); ZeRO owner-gating composes. Under
  `--optim-state-offload` / CPDT moment-precision plans v stays fully
  allocated (their stage-in envelopes touch both moments) with a loud
  note; reduced-precision moments + muon refuse outright.
- **Muon composes with `--layerwise-accum`** (item 11, the
  separate-accumulator full-quality mode): the window backward
  accumulates RAW per-layer gradient sums (Deferred-shaped plan,
  accum_scale forced to 1.0 — the FullBuffer convention) and the
  per-layer group updates dispatch the stdlib `muon_step` (bias
  correction from the micro-batch counter, as the non-layerwise path
  does). Gated BIT-IDENTICAL to the non-layerwise Muon run on CPU and
  GPU (loss stream + saved model bytes). The exact one-buffer
  "classical Muon" accumulation is intentionally NOT this mode and
  would ship separately named.

### Known issues — found by the P1 Muon campaign (not fixed here)

- The plain-call inference path (`m.forward(...)` from functions,
  callbacks, or top-level code — NOT the source-AD-extracted train step)
  LEAKS its device intermediates: a [2, 1024] coder50m forward leaves
  ~2.3 GB of un-freed GPU tensors per call (the non-flash GQA attention
  scores dominate), and `@no_grad` does not change it — the temps are
  simply never freed. Repeated in-training validation forwards OOM'd
  after 3 calls until the campaign switched to [2, 256] eval windows.
  Needs a bounded-lifetime fix for function-body tensor temps.
- Returning from a user fn whose body mixes tensor ops with several
  scalar (`.item()`-derived) `let` locals SIGSEGVs the emitted program
  shortly after the call returns (JIT frames, masked as exit 1 by
  `execute_temp_build`; repro: `shape_probe`'s original fn-based form).
- A float `let` after tensor `let`s inside a top-level `while` body
  fails Cranelift verification: "declared type of variable varN doesn't
  match type of value vM". Both worked around by inlining + avoiding
  scalar locals in `models/benchmarks/muon50m/shape_probe.nsl`.
- Subscripting a model fixed-array field from a CALLBACK body
  (`m.blocks[0].w_up` in `on_step`) compiles but SIGSEGVs the emitted
  program; the `for pb in m.blocks:` iteration form works (used by the
  zero3 callback gate).

### Added — P5: full-precision mixed Muon/AdamW optimizer

- `Muon(...)` in train blocks is now the REAL Muon (Jordan et al., 2024):
  rank-2 hidden weight matrices take the quintic Newton-Schulz
  orthogonalized-momentum update (`ns_steps`, default 5) scaled by
  `sqrt(max(1, rows/cols))`; embeddings and the LM head (name-routed:
  embed/lm_head/unembed/wte/wpe/vocab in the param path) plus all
  non-rank-2 params (biases, norms) take a standard AdamW step
  (`beta1`/`beta2`/`eps`/`weight_decay`). Previously `muon` was plain
  momentum-SGD with decoupled weight decay — out of spec.
- Muon is now two-state (m = Muon momentum / AdamW first moment; v = AdamW
  second moment). The routing table prints at train start (`[muon]` line).
- The Newton-Schulz chain is pinned against an independent f64 reference
  (wide + tall branches), the all-AdamW-routed case is bit-exact vs
  `AdamW(...)`, and GPU training is gated (device f32 end-to-end).
- `@pipeline` train blocks refuse `Muon` loudly (not wired there yet).

### Fixed — ZeRO deferred hardening (post-#404)

- Stage-2 reduce_scatter groups now split into padded sub-groups capped at
  `NSL_ZERO_BUCKET_MB` (fractional MB accepted), and tensors larger than
  cap/ws are chunked by byte range — a 1B-scale gradient group (or a single
  64MB embedding grad) no longer hits the 64MB CPU-shm slot wall.
- Cross-rank partition-plan verification: every rank hashes
  (stage, ws, param sizes, owners) and compares against rank 0 at partition
  time — a divergent plan refuses loudly instead of training a torn model.
- Non-owner `m_partial` cleanup uses a true zero-fill instead of `x *= 0.0`
  (which preserved NaN/Inf forever after a diverging window).
- `cuda_device_name()` probes the device the process will actually bind
  (NSL_CUDA_DEVICE / spawner striping), not unconditionally ordinal 0.
- `NcclBackend` tracks `ncclCommAbort` and skips `ncclCommDestroy` in Drop
  after an abort (destroying an aborted communicator is undefined).

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

### Added — Architecture hardening (cont.: config decomposition, PTX metadata, FFI/state)

- `CompileOptions` decomposition continued: extracted the `csha_*` cluster into
  `CshaOptions` and the `cpdt_*` cluster into `CpdtOptions` (joining the existing
  `WcetOptions`/`ZkOptions`/`WggoOptions`). `CompileOptions.{csha,cpdt}` replace
  six flat fields; behavior-preserving. The `calibration_*` and dev-tools
  clusters are left flat deliberately (already prefix-cohesive; their field names
  collide with identically-named fields on other structs, so a blind rename is
  unsafe) — rationale recorded in `STATUS.md`.
- `nsl_codegen::ptx_metadata` — static, dependency-free, GPU-free parser that
  extracts per-kernel declared register counts, static shared-memory bytes, and
  target SM from synthesized PTX text, plus a report formatter that flags kernels
  exceeding the 255-register per-thread cap. Covered by unit tests + a public-API
  integration test.
- `nsl ptx-metadata <file.ptx>` — CLI subcommand surfacing the per-kernel PTX
  resource report (registers / shared memory / target SM). Pure text analysis;
  no CUDA toolkit required.
- FFI safety tests: `grad_context::abi_layout_tests::magic_is_first_field_for_ffi_validation`
  pins `GradContext.magic` at offset 0 (the only field the C side reads through
  the opaque handle); `c_header_abi_version_matches_runtime_constants` asserts the
  generated header's `NSL_ABI_VERSION_*` macros equal the live runtime constants
  (catches codegen/runtime version skew; no C compiler needed).
- Experimental subsystem feature flags: `experimental-wrga` / `experimental-cpdt`
  (both in `default`) gate the WRGA and CPDT pass entry points in `stmt.rs`, so a
  `--no-default-features` build can turn those research passes into no-ops. The
  default build is byte-identical (gates compile out). Phase-1 behavioral gating;
  see `docs/architecture/compiler-state.md`.
- `docs/architecture/compiler-state.md` — audit of compiler/runtime thread-local
  globals (classified test-only / FFI-OK / migrate), establishing `Compiler` as
  the session object and a staged plan to retire the WRGA build-side globals into
  explicit context (review item: "replace hidden thread-local state").
- `docs/hardware/cuda_status.md` — "Golden CPU-reference test coverage" section
  making the GPU-vs-CPU-oracle validation pattern traceable to specific tests.

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
