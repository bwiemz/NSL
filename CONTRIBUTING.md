# Contributing to NeuralScript

Thank you for your interest in contributing to NeuralScript! This guide explains how to get started.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally: `git clone https://github.com/YOUR_USERNAME/NSL.git`
3. **Create a branch** for your work: `git checkout -b feature/my-feature`
4. **Make your changes** following the guidelines below
5. **Submit a pull request** against the `main` branch

## Development Setup

```bash
# Install Rust (stable)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the project
cargo build

# Run tests
cargo test

# Run linter
cargo clippy
```

## Code Quality Requirements

All contributions must pass before merge:

- **`cargo build`** — zero errors, zero warnings
- **`cargo test`** — all tests pass, no regressions
- **`cargo clippy`** — zero warnings
- **New features** must include unit tests
- **Codegen changes** should include snapshot tests (`insta`)
- **Kernel/fusion changes** should include differential tests

## Pull Request Process

1. All code changes require a **pull request** — direct pushes to `main` are blocked
2. PRs must be **reviewed and approved** by a maintainer before merge
3. Keep PRs focused — one logical change per PR
4. Write a clear PR description explaining **what** and **why**
5. Reference any related issues or milestone numbers (e.g., "M45: Tensor Debugger")

## Commit Messages

Follow the project convention:

```
feat(m45): implement tensor trace recording

Add TraceRecorder with NaN sentinel detection and binary trace format.

Co-Authored-By: Your Name <your@email.com>
```

Prefix: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`

## Architecture

- `crates/nsl-lexer/` — Tokenizer
- `crates/nsl-parser/` — Parser (AST generation)
- `crates/nsl-ast/` — AST types
- `crates/nsl-semantic/` — Type checking, ownership, determinism
- `crates/nsl-codegen/` — Cranelift IR generation, kernel compilation
- `crates/nsl-runtime/` — Runtime library (tensor ops, serving, GPU)
- `crates/nsl-cli/` — CLI entry point (`nsl run`, `nsl build`, etc.)

## Questions?

Open an issue on GitHub for any questions about contributing.
