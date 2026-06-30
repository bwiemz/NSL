# NSL Status

This file is the single source of truth for **which parts of NSL are stable,
which are maturing, and which are research/experimental.** It exists so that a
reviewer, contributor, or potential user can tell at a glance how much to trust
any given subsystem.

NSL is **pre-1.0**. Only the **Stable** tier carries a compatibility
expectation. Beta features work but may change. Experimental features are
research vehicles: they may change shape, regress, or be removed between
releases, and they are not part of the green-build contract.

Last reviewed against: `main` @ v0.9.0 line (2026-06).
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
- **Operator fusion** — automatic elementwise-chain fusion (M31).
- **DataLoader** — zero-copy mmap tokenized-data loading (M19).
- **CLI** — `nsl check`, `nsl run`, `nsl build`, `nsl fmt`, `nsl test`.

**Stable contract (must stay green, gated by CI):**

```bash
cargo build --workspace
cargo clippy --workspace -- -D warnings
```

---

## Beta

Works today, exercised in CI and examples, but still hardening — expect rough
edges and occasional API churn.

- **CUDA / PTX backend** — native GPU codegen and kernel launch. Validated on
  specific hardware (see [`docs/hardware/`](docs/hardware/)); not yet a
  cross-vendor guarantee.
- **Autodiff** — tape-based reverse-mode AD (default) and `--source-ad`
  compile-time lowering with diagnostic fallback.
- **Training DSL** — `train` blocks, optimizers, LR schedulers.
- **Quantization** — FP8, BitNet, AWQ/GPTQ precision tiering (`quantization`).
- **FlashAttention** — codegen path and selector (`analysis`).
- **C ABI / shared-library export** — `nsl_model_*` C API and generated headers
  (M62). See `crates/nsl-runtime/ARCHITECTURE.md` and `docs/abi/`.
- **ONNX import** and **safetensors** loading.

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
- **FASE** — quantization-aware optimizer/codegen.
- **ZK** — zero-knowledge proof generation/verification (`nsl build --zk`).
- **FPGA / Verilog** — HDL backend (Yosys/Verilator nightly job). See
  [`docs/hardware/fpga_status.md`](docs/hardware/fpga_status.md).
- **Unikernel** — `nsl build --unikernel` deployment target.
- **Distributed** — tensor / pipeline / context parallelism, MoE serving.
- **Inference serving** — speculative decoding, paged KV, disaggregated serving.
- **Non-CUDA GPU backends** — AMDGPU/ROCm, Metal, WGSL/WebGPU KIR are built but
  **untested on real hardware**.

---

## How this maps to tests

| Tier         | Required on every PR | Nightly / env-gated         | Informational only |
|--------------|----------------------|-----------------------------|--------------------|
| Stable       | build, clippy, unit  | —                           | —                  |
| Beta         | build                | CUDA, ONNX, e2e (toolchain) | perf baselines     |
| Experimental | build                | FPGA (Yosys/Verilator)      | research compares  |

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the exact commands per tier.
