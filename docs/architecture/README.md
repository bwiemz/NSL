# docs/architecture

Reader-facing descriptions of how each crate works today: pipeline stages,
key types, the invariants the code relies on, the tests that enforce them,
and "where to add a new X" recipes. One document per crate (roadmap Doc1).
Each was written against the code and every path it cites is checked to
exist; when a section stops matching the code, fix the section in the same
PR as the code.

| Document | Covers |
|---|---|
| [`frontend.md`](frontend.md) | `nsl-errors`, `nsl-lexer`, `nsl-ast`, `nsl-parser` — spans and diagnostics, the indentation-aware tokenizer, the AST, recursive descent + Pratt parsing, recovery, the fuzz and round-trip gates |
| [`semantic.md`](semantic.md) | `nsl-semantic` — the pass pipeline, `Type`/`TypeMap`/scopes, compile-time shape checking, the Training Configuration Contract, decorators, effects and ownership, the interface codegen consumes |
| [`codegen.md`](codegen.md) | `nsl-codegen` — entry points and `CompileOptions`, the `Compiler` driver and the train-block compiler, the runtime-ABI registry, KernelIR vs hand-written PTX, autodiff, the pass scheduler, the error model |
| [`runtime.md`](runtime.md) | `nsl-runtime` — the tensor handle model, memory, the autodiff tape, the CUDA backend, training state and checkpoints, data, serving and interop, observability |
| [`cli.md`](cli.md) | `nsl-cli` — the command tree and pipeline, the flag contract (`feature_rules`, `exec_markers`, meta-flags), generated references, and the gate suite |
| [`compiler-state.md`](compiler-state.md) | Where mutable state lives during compilation and at runtime; the thread-local inventory and its migration direction |
| [`2026-08-15-milestone-c-trainir-reassessment.md`](2026-08-15-milestone-c-trainir-reassessment.md) | Why the train block did not get a mid-level IR in Milestone C, and what the `PassManager` took instead |

Two shorter maps live beside the code and are extended, not duplicated, by
the documents above: [`crates/nsl-codegen/ARCHITECTURE.md`](../../crates/nsl-codegen/ARCHITECTURE.md)
(the facade namespaces) and [`crates/nsl-runtime/ARCHITECTURE.md`](../../crates/nsl-runtime/ARCHITECTURE.md)
(the FFI safety contract).

The design history — what each subsystem was designed to do, and the
measurements behind it — is indexed by subsystem in
[`docs/superpowers/README.md`](../superpowers/README.md). The maturity of
each subsystem is in [`STATUS.md`](../../STATUS.md).
