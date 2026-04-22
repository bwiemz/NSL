<!-- owner: @bwiemz -->

# NSL Contributor Wiki

Documentation for developers working **on** NeuralScript. For users of the language, start with the main [README](../../README.md) and [`spec/`](../../spec/).

## Start here

- **[Home](Home.md)** — role-based navigator. Pick your path.
- **[Development-Setup](Development-Setup.md)** — get a working build in under 30 minutes.
- **[Architecture-Overview](Architecture-Overview.md)** — 10,000-ft view of the compiler and runtime.

## Deep dives

- **[Compiler-Pipeline](Compiler-Pipeline.md)** — lexer → parser → semantic → codegen, stage by stage.
- **[Optimization-Passes](Optimization-Passes.md)** — AOT autodiff, memory planning, CSHA / WRGA / WGGO / FASE / CCR.
- **[Runtime-Internals](Runtime-Internals.md)** — `NslTensor`, allocator, GPU path, FFI conventions.

## Working in the repo

- **[Testing-Strategy](Testing-Strategy.md)** — unit, snapshot, differential, e2e.
- **[Adding-a-Language-Feature](Adding-a-Language-Feature.md)** — end-to-end walkthrough using `@export` from M62.

## Reference

- **[Roadmap](Roadmap.md)** — M9 → M62, phases, what's in flight.
- **[Glossary](Glossary.md)** — every acronym, every keyword, every decorator.
- **[Syntax-Reference](Syntax-Reference.md)** — cheat sheet (full reference lives in [`spec/`](../../spec/)).
- **[Examples-Guide](Examples-Guide.md)** — curated reading order for [`examples/`](../../examples/).

---

*Design: [`docs/superpowers/specs/2026-04-21-nsl-contributor-wiki-design.md`](../superpowers/specs/2026-04-21-nsl-contributor-wiki-design.md) (local-only per `.gitignore`).*
