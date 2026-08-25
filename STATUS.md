# NSL Status

This file is the single source of truth for **which parts of NSL are stable,
which are maturing, and which are research/experimental.** It exists so that a
reviewer, contributor, or potential user can tell at a glance how much to trust
any given subsystem.

NSL is **pre-1.0**. Only the **Stable** tier carries a compatibility
expectation. Beta features work but may change. Experimental features are
research vehicles: they may change shape, regress, or be removed between
releases, and they are not part of the green-build contract.

Last reviewed against: `main` @ v0.9.0 line (2026-08-25, item-19 reconciliation).
No v0.9.1 was ever tagged; everything since the v0.9.0 tag (2026-03-19) is
unreleased work on the 0.9 line.
If you change a subsystem's maturity, update this file in the same PR.

---

## Tier definitions

| Tier             | Meaning                                                                                  | CI expectation                              |
|------------------|------------------------------------------------------------------------------------------|---------------------------------------------|
| **Stable**       | Part of the core language contract. Breaking changes need a deliberate, documented bump. | Must build + clippy clean + unit tests pass |
| **Beta**         | Works and is used, but hardening, API, and coverage are still in motion.                  | Built in CI; some tests env-gated           |
| **Experimental** | Research subsystem. Opt-in. No stability promise. Validated by research tests only.       | Build-gated only; research tests informational |

The crate-level facade namespaces in `nsl-codegen` (`core`, `gpu`, `training`,
`quantization`, `distributed`, `analysis`, `experimental`) mirror these tiers;
`experimental::*` is the Experimental tier by definition.

---

## Stable

The boring, must-always-work core.

- **Frontend** — `nsl-lexer`, `nsl-ast`, `nsl-parser` (indentation-aware
  tokenizer, recursive-descent parser).
- **Semantic analysis** — type checking, name resolution, **compile-time tensor
  shape checking** (`nsl-semantic`).
- **CPU codegen** — Cranelift IR lowering and native object emission for the CPU
  target (`nsl-codegen` `core`).
- **CPU runtime** — core tensor ops, memory management, autograd tape
  (`nsl-runtime` tensor/memory/autodiff on CPU).
- **Operator fusion** — automatic elementwise-chain fusion (M31). The
  elementwise-chain core is the stable part; broader graph-rewrite integration
  is still uneven (see `docs/summaries/06-implementation-status.md`) and is
  NOT covered by the stable contract.
- **DataLoader** — zero-copy mmap tokenized-data loading (M19).
- **CLI** — `nsl check`, `nsl run`, `nsl build`, `nsl fmt`, `nsl test` carry
  the stability promise. The full shipped surface is larger (`export, convert,
  init, debug, zk, profile, autotune, tokenize, fpga-compile, ptx-metadata,
  stats, prove, verify`) — those ride their subsystem's tier, not this one.

**Compatibility contract:** only the Stable tier above carries a cross-version
"won't break" promise. That promise is *narrower* than the **CI merge gate** —
CI blocks every PR on more than just the Stable tier (build, workspace unit
tests, clippy, the CLI e2e suite on Linux/Windows, and the ONNX-RT and FPGA
jobs). `.github/workflows/ci.yml` is the source of truth; the table at the
bottom of this file maps each tier to what CI runs.

```bash
# The floor every contributor can reproduce locally:
cargo build --workspace
cargo clippy --workspace -- -D warnings
cargo test --workspace -- --skip e2e_
```

---

## Beta

Works today, exercised in CI and examples, but still hardening — expect rough
edges and occasional API churn.

- **CUDA / PTX backend** — native GPU codegen and kernel launch. Validated on
  specific hardware (see [`docs/hardware/`](docs/hardware/)); not yet a
  cross-vendor guarantee.
- **Autodiff** — tape-based reverse-mode AD (default) and `--source-ad`
  compile-time lowering. Source AD REFUSES what it cannot lower soundly
  (unresolved dropout probability, unresolved Input leaves) rather than
  falling back silently; `distill` refuses the tape fallback outright.
  Dropout under source AD carries the exact forward mask (P0 fix, 2026-08).
- **Training DSL** — `train` blocks, optimizers, LR schedulers. The config
  namespace is CLOSED (unknown/duplicate/non-literal keys are compile errors
  — the Training Configuration Contract), and training state is resumable:
  the `.optim` sidecar v2 restores optimizer moments, data position, and every
  RNG stream, refusing on corpus/geometry drift (item 8).
- **Quantization** — FP8, BitNet, AWQ/GPTQ precision tiering (`quantization`).
- **FlashAttention** — codegen path and selector (`analysis`).
- **C ABI / shared-library export** — `nsl_model_*` C API and generated headers
  (M62), including the DLPack output-ownership models `nsl_model_call_into` /
  `nsl_model_call_alloc` + `nsl_model_get_export_signature` (item 7). See
  `crates/nsl-runtime/ARCHITECTURE.md` and `docs/abi/`.
- **ONNX import** and **safetensors** loading.
- **Structured runtime events** — `NSL_EVENTS=<path>` JSONL stream twinning
  the counter markers + per-step GPU memory with exact bytes (item 17);
  stderr stays byte-identical.
- **Pretokenization pipeline** — two-stage BPE tokenizers of record
  (`models/tokenizers/`), u16 corpus format, and the fast byte-domain encoder
  (`tokbench --backend fast`, same-token parity gated; items 15/16).

---

## Experimental

Research subsystems. **Opt-in, no stability promise.** Exercised by research
tests that are *not* part of the green-build contract (see README → Benchmarks
→ test tiers). Most live under `experimental::*` in `nsl-codegen` /
`nsl-runtime`.

- **WGGO** — weight-graph global optimization.
- **WRGA** — weight-rewrite / gated-LoRA adapter codegen (PEFT fusion).
- **CEP** — compiler-extracted pruning.
- **CFIE** — fused inference engine (speculative, grammar, KV planning).
- **CSHA** — compiler-specialized hardware attention (FlashAttention-v2 tiers).
- **CPDT** — compiler-planned distributed training (ZeRO, expert/precision tiers).
- **CPKD** — compiler-planned knowledge distillation (distill block, frozen
  teacher, fused KL-CE GPU kernel).
- **FASE** — quantization-aware optimizer/codegen.
- **ZK** — zero-knowledge proofs: the folding backend is end-to-end
  (`nsl build --zk-backend folding` + `nsl zk verify`); halo2/plonky3 are
  refused at compile time.
- **FPGA / Verilog** — HDL backend (Yosys/Verilator nightly job). See
  [`docs/hardware/fpga_status.md`](docs/hardware/fpga_status.md).
- **Unikernel** — `nsl build --unikernel` deployment target.
- **Distributed** — tensor / pipeline / context parallelism, MoE serving.
- **Inference serving** — speculative decoding, paged KV, disaggregated serving.
- **Non-CUDA GPU backends** — AMDGPU/ROCm, Metal, WGSL/WebGPU KIR are built but
  **untested on real hardware**.
- **SR-BF16 parameter storage** — `--param-dtype bf16-sr` (stochastic
  rounding). Mechanism + refusals complete; a real-corpus differential found
  no quality delta, and f32 remains the default.
- **CUDA graph capture** — `--cuda-graphs` whole-step capture; several
  features taint capture and fall back.

---

## Opting out of experimental subsystems

The experimental passes are compiled in by default. A downstream build can opt
out of a research subsystem with Cargo features (phase-1: behavioral gating at
the pass entry point — the pass becomes a no-op; it does not yet strip the
modules from the binary):

```bash
# Build with WRGA + CPDT planning disabled (passes become no-ops):
cargo build -p nsl-codegen --no-default-features --features "<keep these>"
```

Currently gated at their entry point: `experimental-wrga`, `experimental-cpdt`
(in `crates/nsl-codegen/Cargo.toml`, both in `default`). WGGO/CSHA/ZK/FPGA
follow the same pattern as gating is extended. See
[`docs/architecture/compiler-state.md`](docs/architecture/compiler-state.md)
for the compiler-state model (and the thread-local audit + migration plan that
the same hardening pass produced).

The `CompileOptions` "god-config" is being decomposed into cohesive sub-structs
(`WcetOptions`, `ZkOptions`, `WggoOptions`, `CshaOptions`, `CpdtOptions`, …).
The `calibration_*` and dev-tools (`profile_*`/`health_*`) clusters are left
flat *deliberately*: they are already prefix-cohesive and their field names
(`target_gpu`, `dtype`, `calibration_data`) collide with identically-named
fields on other structs, so a mechanical rename is unsafe without per-site type
analysis. Group them only alongside that analysis.

## How this maps to tests

| Tier         | CI gate (blocks every PR)                            | Informational / non-blocking              |
|--------------|------------------------------------------------------|-------------------------------------------|
| Stable       | build, clippy, workspace unit tests (`--skip e2e_`)  | —                                         |
| Beta         | CLI e2e (Linux/Windows), ONNX-RT integration job     | real-CUDA-device tests, perf baselines    |
| Experimental | `fpga` job (build/lint gate)                         | FPGA Verilator/Yosys **nightly** workflow, `#[ignore]`'d research tests, macOS e2e |

CI jobs are cumulative — the Beta/Experimental rows run *in addition to* the
Stable row on every PR (they are separate, blocking CI jobs, not nightly).
Four drift gates also block every PR outside the tier table: `version-agreement`
(Cargo == spec/README/CLI/C API/python), `doc-agreement` (docs == tree),
`gpu-gate-inventory` (cert-lane manifest == tree), and `python-interop` +
`cuda-feature` (compile + GPU-free tests). GPU execution itself is certified by
the LOCAL lane (`scripts/gpu-tier.sh smoke|certify|endurance`, item 14) on the
self-hosted box — GitHub runners have no GPU.
Non-blocking entries are `continue-on-error` jobs or tests CI cannot run (no GPU
on the runners). See [`CONTRIBUTING.md`](CONTRIBUTING.md) and
`.github/workflows/ci.yml` for the exact commands.
