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

**Required on every PR (CI merge gates — `.github/workflows/ci.yml` is the
source of truth):**

```bash
cargo build --workspace                              # zero errors, zero warnings
cargo clippy --workspace -- -D warnings              # zero warnings
cargo test --workspace -- --skip e2e_                # workspace unit/integration tests
cargo test -p nsl-cli --test e2e -- --test-threads=1 # CLI smoke/e2e (Linux + Windows)
```

The ONNX Runtime (`test-onnx-rt`) and FPGA Verilator/Yosys (`fpga`) CI jobs
**also block every PR** — they run in dedicated jobs rather than the command
list above, but a red one blocks merge just the same. The e2e step is
non-blocking only on macOS (that matrix is `continue-on-error`); locally it
needs a full toolchain (C linker, optional OpenSSL/CUDA), so a missing-toolchain
failure on your machine is a local-environment gap, not a relaxed gate.

**Required for the area you touch:**

- **New features** must include unit tests.
- **Codegen changes** should include snapshot tests (`insta`).
- **Kernel/fusion changes** should include differential tests.

**Non-blocking (informational):**

- the macOS matrix + its e2e step (`continue-on-error`),
- real-CUDA-device tests (CI runners have no GPU) and `#[ignore]`'d research
  tests in the `experimental::*` subsystems (CEP, CFIE, CSHA, WGGO, WRGA, CPDT,
  ZK, FPGA, …), plus performance-baseline comparisons.

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

## Review gates (what reviewers will push back on)

To keep the core stable while research subsystems evolve, reviewers hold the
line on a few structural rules. Save a round-trip by checking these before you
open a PR:

- **Stable vs experimental:** new research passes/subsystems land in the
  `experimental::*` namespace and must be **opt-in** (not enabled by default).
  Update [`STATUS.md`](STATUS.md) in the same PR.
- **Runtime FFI:** new exported C-ABI symbols must document ownership,
  nullability, alignment, length, error behavior, threading, and the no-unwind
  rule (see [`docs/abi/`](docs/abi/)). Host-facing symbols go in the generated
  header; layout changes to `NslTensorDesc` are a **major** ABI bump.
- **Hardware claims:** a new "supported on hardware X" claim needs a golden
  correctness test (vs CPU reference) and a row in [`docs/hardware/`](docs/hardware/).
  Don't claim in docs what CI doesn't validate.
- **Config sprawl:** prefer grouping over adding yet another top-level field to
  `CompileOptions`; prefer a subcommand/module over piling logic into
  `nsl-cli/src/main.rs`.
- **Clippy suppressions:** fix or locally `#[allow(...)]` at the site; don't
  broaden crate-wide suppressions to get a build green.

## Architecture

For architectural documentation (crate graph, compiler pipeline, runtime internals, optimization passes, how to add a language feature), see [`docs/wiki/Architecture-Overview.md`](docs/wiki/Architecture-Overview.md).

For the runtime C-ABI contract and FFI safety rules, see [`docs/abi/`](docs/abi/).
For the stable/beta/experimental tiering of every subsystem, see [`STATUS.md`](STATUS.md).

## Questions?

Open an issue on GitHub for any questions about contributing.
