<!-- owner: @bwiemz -->

# Architecture Overview

10,000-ft view of what happens when you run `nsl build main.nsl`. Deep-dives live in [Compiler-Pipeline](Compiler-Pipeline.md), [Optimization-Passes](Optimization-Passes.md), and [Runtime-Internals](Runtime-Internals.md).

## The stack at a glance

```text
┌────────────────────────────────────────────────────────────────────┐
│  main.nsl (source)                                                 │
└─────────┬──────────────────────────────────────────────────────────┘
          │
          ▼
   ┌──────────────┐
   │   nsl-lexer  │  tokens
   └──────┬───────┘
          ▼
   ┌──────────────┐     ┌──────────────┐
   │  nsl-parser  │────►│   nsl-ast    │  AST node types (shared contract)
   └──────┬───────┘     └──────────────┘
          ▼
   ┌──────────────┐
   │ nsl-semantic │  shape/ownership/determinism checks
   └──────┬───────┘
          ▼
   ┌──────────────┐                        ┌────────────────────────────────┐
   │ nsl-codegen  │── Cranelift IR ──────► │ native binary                  │
   │              │                        │  • machine code                │
   │              │── PTX text bytes ────► │  • embedded PTX (data section) │
   └──────┬───────┘                        └──────────────┬─────────────────┘
          │                                               │ cuModuleLoadData
          ▼                                               ▼
    ┌──────────────┐                                ┌──────────────┐
    │  nsl-runtime │  tensor ops, allocator         │  CUDA driver │  (at runtime)
    └──────────────┘                                └──────────────┘

   ┌──────────────┐   ┌──────────────┐
   │  nsl-errors  │   │   nsl-cli    │  nsl run / build / profile / check
   └──────────────┘   └──────────────┘
```

## Crates at a glance

| Crate | Responsibility |
|---|---|
| [`nsl-errors`](../../crates/nsl-errors/) | Diagnostic formatting — true leaf, no internal deps |
| [`nsl-runtime`](../../crates/nsl-runtime/) | Runtime library (tensor ops, serving, GPU, allocator) — no internal deps |
| [`nsl-ast`](../../crates/nsl-ast/) | AST node types — shared data contract; depends only on `nsl-errors` |
| [`nsl-lexer`](../../crates/nsl-lexer/) | Tokenization, indentation handling; depends on `nsl-errors` |
| [`nsl-parser`](../../crates/nsl-parser/) | Parsing, AST construction, error recovery; depends on `nsl-lexer`, `nsl-ast`, `nsl-errors` |
| [`nsl-semantic`](../../crates/nsl-semantic/) | Type inference, shape checking, ownership analysis; depends on `nsl-ast`, `nsl-lexer`, `nsl-parser`, `nsl-errors` |
| [`nsl-codegen`](../../crates/nsl-codegen/) | Cranelift IR emission for all functions; separate PTX synthesis pipeline for GPU kernels; optimization passes; depends on all above plus `nsl-runtime` |
| [`nsl-cli`](../../crates/nsl-cli/) | `nsl run`, `nsl build`, `nsl profile`, `nsl check`; depends on all other crates |

Dependency direction: `nsl-errors` and `nsl-runtime` are the true leaves (nothing they depend on is NSL-internal). Everything else builds upward. `nsl-cli` is the root and pulls in all seven other crates.

> **Note on `nsl-ast`:** the AST node types depend on `nsl-errors` (for span/diagnostic types), so `nsl-ast` is not a pure leaf — but it has no dependency on any other NSL crate higher in the pipeline.

## What lives where

- **Compiler & runtime primitives**: Rust. Fast to build, no GC, direct cudarc / Cranelift bindings.
- **Standard library** (`nsl.nn`, `nsl.optim`, losses, tokenizers): [NSL itself](../../stdlib/). The language is self-hosting for everything that isn't compile-time or hardware-adjacent.
- **Examples & tests**: `.nsl` files under [`examples/`](../../examples/) and [`tests/`](../../tests/) drive compile + run smoke coverage.

## GPU layer

NSL binds to NVIDIA CUDA via [cudarc](https://github.com/coreylowman/cudarc) (0.19, dynamic linking). Cranelift is the sole function-emission backend; **PTX is synthesized separately and embedded as Cranelift data sections**, then loaded at runtime via `cuModuleLoadData`. The portable PTX emitter lives in [`crates/nsl-codegen/src/backend_ptx.rs`](../../crates/nsl-codegen/src/backend_ptx.rs) (fed by [`kernel_lower.rs`](../../crates/nsl-codegen/src/kernel_lower.rs)); a legacy direct-AST-to-PTX path lives in [`kernel.rs`](../../crates/nsl-codegen/src/kernel.rs) for kernels outside the portable subset. See [Compiler-Pipeline § Stage 4](Compiler-Pipeline.md#stage-4--codegen) for the full picture.

## Subsystem deep-dives

The non-obvious compiler sophistication lives in optimization passes — they're what make NSL faster than a naive PyTorch port. See [Optimization-Passes](Optimization-Passes.md) for:

- [CSHA](Glossary.md#csha) — compile-time static hybrid attention
- [WRGA](Glossary.md#wrga) — weight-rank graph analysis / adapter fusion
- [CPDT](Glossary.md#cpdt) — weight-aware compilation / tensor parallelism
- [FASE](Glossary.md#fase) — training-loop fusion
- [WGGO](Glossary.md#wggo) — graph optimization at the weight-gradient level
- [CCR](Glossary.md#ccr) — common-kernel combination rewriting

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21. If the crate graph or pass order in this page no longer matches reality, open an issue tagged `docs-rot`.*
