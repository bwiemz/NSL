<!-- owner: @bwiemz -->

# Home

Welcome to the NSL contributor wiki.

NSL is a statically-typed, compiled programming language designed as a first-class replacement for Python + PyTorch in AI/ML workloads. It compiles to native code via Cranelift with zero Python or C++ dependencies — the entire stack is Rust (compiler + runtime) and NSL (standard library). NSL features compile-time tensor shape checking, native autodiff, a declarative `train` block DSL for training workflows, and GPU/CUDA native kernels for custom operations.

## I am a…

### …new contributor
Start here:
1. [Development-Setup](Development-Setup.md) — get a working build
2. [Architecture-Overview](Architecture-Overview.md) — 10,000-ft view
3. [Adding-a-Language-Feature](Adding-a-Language-Feature.md) — worked example

Then pick a milestone from [Roadmap](Roadmap.md) or an open issue.

### …compiler hacker
1. [Compiler-Pipeline](Compiler-Pipeline.md) — lexer → parser → semantic → codegen
2. [Optimization-Passes](Optimization-Passes.md) — IR rewrites, memory planning
3. [Testing-Strategy](Testing-Strategy.md) — how to prove your change works

### …runtime / GPU dev
1. [Runtime-Internals](Runtime-Internals.md) — `NslTensor`, allocator, CUDA path
2. [Optimization-Passes § memory planning](Optimization-Passes.md#memory-planning--the-slab-allocator) — slab model
3. [Examples-Guide § subsystem demos](Examples-Guide.md#subsystem-demos) — kernel reference files

### …ML user evaluating NSL
1. Main [README](../../README.md) — install + run in 60 seconds
2. [Syntax-Reference](Syntax-Reference.md) — cheat sheet
3. [Examples-Guide](Examples-Guide.md) — curated reading order
4. [Roadmap](Roadmap.md) — what's shipping next

## What's in flight right now

- **M52 CPDT Phase 2** — weight-aware spectral factor measurement-triggered at >20% committed-fixture disagreement ([Roadmap](Roadmap.md))
- **M62 PyTorch FFI** — per-function C wrappers and Python E2E tests remain ([`m62-c-wrappers-design.md`](../superpowers/specs/2026-04-15-m62-c-wrappers-design.md))
- **AWQ retention subprocess gaps** — model library linking + model_forward calls ([`docs/plans/`](../plans/))
- **WGGO Phase 2 gradient scoring** — backward-pass execution blocked on AWQ retention fix ([Roadmap](Roadmap.md))
- **WRGA B.3.2 fused backward** — scope analysis in progress; speedup estimate 1.2–1.5x ([`2026-04-18-wrga-b32-fused-backward-STUB.md`](../plans/2026-04-18-wrga-b32-fused-backward-STUB.md))

## If you get stuck

- Acronyms: [Glossary](Glossary.md)
- PR / commit conventions: [`CONTRIBUTING.md`](../../CONTRIBUTING.md)
- Design artifacts: `docs/plans/`, `docs/superpowers/specs/`, `docs/research/`, `docs/summaries/` (mostly local-only per `.gitignore`; ask in issues)
- Memory files: agent-private (`~/.claude/projects/`). Not intended for PR flow.

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21. If the crate graph or pass order in this page no longer matches reality, open an issue tagged `docs-rot`.*
