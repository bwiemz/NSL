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

NSL separates tests into tiers so the stable contract stays meaningful. See
[`STATUS.md`](STATUS.md) for what is Stable vs Beta vs Experimental.

**Required on every PR (the stable green-build contract):**

```bash
cargo build --workspace                      # zero errors, zero warnings
cargo clippy --workspace -- -D warnings      # zero warnings
cargo test --workspace -- --skip e2e_        # workspace unit/integration tests
```

**Required for the area you touch:**

- **New features** must include unit tests.
- **Codegen changes** should include snapshot tests (`insta`).
- **Kernel/fusion changes** should include differential tests.

**Nightly / environment-gated (not a per-PR merge blocker):**

```bash
cargo test -p nsl-cli --test e2e -- --test-threads=1   # needs full toolchain (C linker, optional OpenSSL/CUDA)
# CUDA, ONNX, and FPGA (Verilator/Yosys) suites run in dedicated CI jobs
```

**Informational / research (Experimental tier — not a merge blocker):**

- `experimental::*` subsystem tests (CEP, CFIE, CSHA, WGGO, WRGA, CPDT, ZK,
  FPGA, …) and performance-baseline comparisons. These may fail depending on the
  checkout/environment; treat results against the tier they belong to.

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

For architectural documentation (crate graph, compiler pipeline, runtime internals, optimization passes, how to add a language feature), see [`docs/wiki/Architecture-Overview.md`](docs/wiki/Architecture-Overview.md).

## Questions?

Open an issue on GitHub for any questions about contributing.
