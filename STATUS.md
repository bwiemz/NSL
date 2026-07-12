# NSL Status

This file is the single source of truth for **which parts of NSL are stable,
which are maturing, and which are research/experimental.** It exists so that a
reviewer, contributor, or potential user can tell at a glance how much to trust
any given subsystem.

NSL is **pre-1.0**. Only the **Stable** tier carries a compatibility
expectation. Beta features work but may change. Experimental features are
research vehicles: they may change shape, regress, or be removed between
releases, and they are not part of the green-build contract.

Last reviewed against: `main` @ v0.9.0 line (2026-07).
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
- **CPKD** — compiler-planned knowledge distillation (distill block, frozen
  teacher, fused KL-CE GPU kernel).
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
| Experimental | FPGA Verilator/Yosys job                             | `#[ignore]`'d research tests, macOS e2e   |

CI jobs are cumulative — the Beta/Experimental rows run *in addition to* the
Stable row on every PR (they are separate, blocking CI jobs, not nightly).
Non-blocking entries are `continue-on-error` jobs or tests CI cannot run (no GPU
on the runners). See [`CONTRIBUTING.md`](CONTRIBUTING.md) and
`.github/workflows/ci.yml` for the exact commands.
