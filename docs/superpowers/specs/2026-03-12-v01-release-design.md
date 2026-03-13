# NeuralScript v0.1 Release Design Spec

## Goal

Ship NeuralScript v0.1.0 as a "Polished MVP" targeting early-adopter ML developers evaluating NSL for real projects. The release must provide a professional first impression: working end-to-end examples, reliable tooling, proper packaging, and clear documentation of both capabilities and known limitations.

## Audience

Early-adopter developers who want to evaluate NSL for training small models (GPT-2 scale), explore the language's ML-native features, and compare with PyTorch. They expect:
- A clean install experience (download, extract, run)
- Working examples they can modify and learn from
- Basic tooling (`fmt`, `init`) that signals a real project
- Clear documentation of what works and what doesn't yet

## Current State

All core milestones M9-M19 are complete:
- M9: Rust runtime + tensor foundation
- M10: Compile-time tensor shape checking
- M11: `model` keyword + layer types
- M12: `grad` keyword + tape-based autodiff
- M13: Import system + multi-file compilation
- M14: Training DSL + 6 optimizers + 7 schedulers
- M15: NN stdlib + BPE tokenization + `@test` framework
- M16: Quantization (INT4/INT8, per-tensor/channel/group)
- M17: GPU/CUDA + `kernel` keyword + 15 PTX kernels
- M18a: Transformer foundations (unsqueeze, expand, stack, causal_mask)
- M18b: Interop (safetensors, HuggingFace loading, ONNX export)
- M19: Data pipeline (JSONL/CSV/mmap, DataLoader, inference sampling)

57 unit tests passing, 47 example programs, 22 stdlib modules, ~36K Rust LOC.

## Identified Gaps

### Blockers
1. **M19 not merged** — complete in `feature/m19-data-pipeline` branch, needs merge to main
2. **Interop linker failure** — examples crash with unresolved symbols (`nsl_safetensors_load`, `nsl_hf_load`, `nsl_onnx_export`) because the `interop` feature flag isn't propagated through the dependency chain (nsl-cli → nsl-codegen → nsl-runtime)
3. **License mismatch** — Cargo.toml says MIT, LICENSE file says Apache 2.0
4. **Missing Cargo.toml metadata** — no authors, repository, homepage, description across crates

### Missing Tooling
5. **No `nsl fmt`** — no code formatter
6. **No `nsl init`** — no project scaffolding

### Quality Gaps
7. **No E2E test automation** — 21 expected output files exist but no harness runs them
8. **90 clippy warnings** — mostly in nsl-runtime, style/optimization issues
9. **Broken examples** — some .nsl files may fail due to the linker issue or stale code

### Release Infrastructure
10. **No CI/CD** — no GitHub Actions workflows
11. **No release artifacts** — no tagged releases, no binary distribution
12. **No crates.io publishing** — crate metadata incomplete
13. **Runtime distribution** — `nsl` binary alone is useless without `libnsl_runtime.a`

### Documentation
14. **README needs refresh** — installation instructions, known limitations
15. **Examples need curation** — milestone prefixes are opaque to newcomers
16. **Error messages** — codegen errors lack source locations

---

## Section 1: Blockers

### 1.1 Merge M19

Merge `feature/m19-data-pipeline` into main. This branch is complete with data loading, sequence packing, and inference sampling. Note: M19 is in a git worktree at `.worktrees/m19-data-pipeline/` — clean up the worktree after merge.

### 1.2 Fix Interop Linker Failure

The `interop` feature must be enabled through the dependency chain: nsl-cli enables `nsl-codegen/interop`, which enables `nsl-runtime/interop`. Make `interop` a default feature on nsl-runtime so it's always linked. This pulls in safetensors, hf-hub, prost, half, and regex-lite — acceptable binary size tradeoff for v0.1 since early adopters expect interop to Just Work. Without this, `nsl run` fails on any program that touches interop (even indirectly through stdlib).

### 1.3 Fix License

Decision: **Apache 2.0**. Update `Cargo.toml` workspace metadata from `license = "MIT"` to `license = "Apache-2.0"`. Verify the LICENSE file at repo root is the correct Apache 2.0 text.

### 1.4 Polish Cargo.toml Metadata

Add to workspace `[package]` and propagate to all crates:
- `authors = ["Brandon Wiemer"]`
- `repository = "https://github.com/bwiemz/NSL"`
- `homepage = "https://github.com/bwiemz/NSL"`
- `keywords = ["ml", "compiler", "tensor", "neural-network", "language"]`
- `categories = ["compilers", "science"]`
- Per-crate `description` fields

---

## Section 2: New Tooling

### 2.1 `nsl fmt`

A token/line-based code formatter for `.nsl` files. NOT an AST-based reformatter — operates on the token stream from the existing lexer.

**Formatting passes:**
- **Indentation safety**: Tab-only files convert to 4 spaces. If mixed tabs+spaces are detected within the same block scope, **hard fail** with error: `error: ambiguous mixed indentation at line X. Fix manually before formatting.` The formatter never guesses block level in an indentation-sensitive language.
- **Operator spacing**: Normalize whitespace around binary operators (`a+b` → `a + b`)
- **Trailing whitespace**: Strip trailing spaces/tabs from all lines
- **Blank lines**: Enforce max 2 consecutive blank lines
- **String quotes**: Normalize to double quotes
- **Import sorting**: Stdlib imports first, then local imports, alphabetized within each group
- **`--check` flag**: Exit non-zero if changes needed (for CI integration)
- **Default behavior**: `nsl fmt file.nsl` writes in-place (like rustfmt/gofmt/black). `--check` is the dry-run mode.

Note: Import sorting requires light parsing beyond raw tokens (understanding import statement structure). Operator spacing must distinguish binary operators from unary minus. These are "light parsing" tasks — not a full AST, but more than pure token streaming. If import sorting proves complex, defer to v0.1.1.

After implementing `nsl fmt`, run it on all stdlib and example files as a reformatting pass. This is an implicit work item — existing files will need formatting updates.

### 2.2 `nsl init`

ML-flavored project scaffolding.

```
nsl init myproject
```

Creates:
```
myproject/
  main.nsl            # Starter with tensor example (not just hello world)
  nsl.toml            # Project config (name, version, entry point)
  data/               # Convention for datasets
  weights/            # Convention for model checkpoints
  .gitignore
```

The `.gitignore` covers both build and ML artifacts:
```gitignore
# Build
*.exe
*.o
.nsl-cache/

# ML artifacts
*.safetensors
*.bin
*.nslm
/weights/
/data/
```

The `nsl.toml` establishes the convention for future package management (v0.2) but is not consumed by the compiler in v0.1. Include a comment in the generated file: `# Reserved for future use by the NSL package manager (v0.2)`.

---

## Section 3: Quality & Testing

### 3.1 E2E Smoke Tests

A Rust test harness that compiles and runs ~20 example files via `nsl run`, captures stdout, and compares against expected baselines in `tests/expected/`.

**Numeric-aware diffing** to prevent float drift flakiness:
1. **Float normalization**: Regex matches floating-point numbers (`-?\d+\.\d+`) and truncates to 4 decimal places before comparison. Baselines are stored pre-truncated. This handles f32/f64 precision differences across platforms.
2. **Path normalization**: Replace absolute paths (`C:\Users\...` or `/home/...`) with `<PATH>` before comparison.

**Test matrix:**
- Basic: hello.nsl, features.nsl
- Tensors: m9_tensors, m10_shape_check, m10_symbolic_dims
- Models: m11_model_basic, m11_model_compose
- Autodiff: m12_grad_basic, m12_grad_matmul
- Imports: m13_stdlib_import
- Training: m14_sgd_basic, m14_adam_scheduler
- Stdlib: m15_tiny_lm, m15_nn_stdlib
- Quantization: m16_quantize
- Interop: m18b_interop
- GPU tests: gated behind CUDA availability check — emit `SKIP: No CUDA device found` rather than panicking

### 3.2 Clippy Cleanup

Fix 90 clippy warnings across all crates. Mostly mechanical: feature-gate unused CUDA constants, simplify redundant closures, use std math equivalents.

### 3.3 Verify All Examples Compile

After fixing the linker issue, systematically run every `.nsl` file in `examples/`. Each must either succeed or fail with a clear user-facing error (not a linker crash or Cranelift panic). Fix any that are broken.

---

## Section 4: CI/CD & Release Infrastructure

### 4.1 GitHub Actions CI

**`ci.yml`** — runs on every push and PR:
- `cargo build --workspace`
- `cargo test --workspace`
- `cargo clippy --workspace -- -D warnings`
- E2E smoke tests (GPU tests skip gracefully on standard runners)
- Matrix: `ubuntu-latest`, `windows-latest`, `macos-14` (Apple Silicon), `macos-13` (Intel)

**Known gap**: Standard CI runners have no GPUs, so CUDA functionality is never tested in CI. GPU tests must skip gracefully (not panic). Manual GPU testing on a CUDA machine is required before each release tag.

### 4.2 Release Workflow & Artifacts

**`release.yml`** — triggered by `v*` tag:
- Build release toolchain bundles for 4 targets
- Create GitHub Release with archives attached
- Publish library crates to crates.io

**Toolchain bundles** (not standalone binaries — the `nsl` compiler requires `libnsl_runtime.a` to link user programs AND `stdlib/` for import resolution):

```
nsl-v0.1.0-x86_64-unknown-linux-gnu.tar.gz
  bin/nsl
  lib/libnsl_runtime.a
  lib/stdlib/nsl/nn/*.nsl
  lib/stdlib/nsl/optim/*.nsl
  lib/stdlib/nsl/data/*.nsl
  lib/stdlib/nsl/tokenize/*.nsl
  lib/stdlib/nsl/inference/*.nsl
  lib/stdlib/nsl/quant/*.nsl
  lib/stdlib/nsl/compat.nsl
  lib/stdlib/nsl/math.nsl
  LICENSE
  README.md

(Same structure for all 4 platform targets)
```

**Linker discovery**: Update `linker.rs` to find `libnsl_runtime.a` relative to the `nsl` executable path (`current_exe()/../lib/`), with fallback to cargo target directory for development builds.

**Stdlib discovery**: Update the import resolver to find stdlib modules at `current_exe()/../lib/stdlib/`, with fallback to the source tree `stdlib/` for development builds.

**Note**: Without the `lib/stdlib/` directory, any program using `from nsl.nn import ...` or similar stdlib imports will fail for end users.

### 4.3 Crates.io Publishing

Publish **library crates only** in dependency order: nsl-errors → nsl-lexer → nsl-ast → nsl-parser → nsl-semantic → nsl-runtime → nsl-codegen (nsl-codegen depends on nsl-runtime, so nsl-runtime must be published first). This is for Rust developers who want to embed the NSL compiler programmatically.

Do **NOT** publish nsl-cli as a `cargo install` target in v0.1 — it won't work without the companion `lib/` directory. End-user install path is GitHub Releases exclusively. Revisit in v0.2 with `include_bytes!()` runtime embedding.

### 4.4 CHANGELOG

`CHANGELOG.md` at repo root in Keep a Changelog format. Single `v0.1.0` entry covering all M9-M19 features organized by category. GitHub Release body is a condensed highlights version.

---

## Section 5: Documentation & Polish

### 5.1 README Refresh

- **Installation**: "Extract the entire `nsl-v0.1.0-<platform>` folder to a permanent location and add its `bin/` directory to your PATH. **Do not separate the `nsl` binary from the `lib/` directory** — the compiler needs `lib/libnsl_runtime.a` to link your programs and `lib/stdlib/` for standard library imports."
- **Prerequisites**: Note that a C linker must be available. On Linux/macOS this is typically `cc`/`gcc`/`clang` (installed by default or via `build-essential`/Xcode CLI tools). On Windows, Visual Studio Build Tools (MSVC `link.exe`) is required.
- **Version check**: `nsl --version` prints the version (add this via clap)
- **Quick start**: `nsl init myproject && cd myproject && nsl run main.nsl`
- **Feature highlights** with short code snippets
- **Known limitations**:
  - No package manager or dependency resolution
  - No PyTorch FFI (`to_torch`/`from_torch`) — use HF loading + ONNX export instead
  - No distributed multi-GPU training (DDP)
  - No REPL
  - CUDA required for GPU features (no ROCm/Metal yet)

### 5.2 Example Curation

Add `examples/README.md` organizing the 47 files by topic rather than milestone:
- Getting Started: hello.nsl
- Tensor Operations: m9_tensors.nsl, m10_shape_check.nsl
- Building Models: m11_model_basic.nsl, m18_transformer.nsl
- Training: m14_sgd_basic.nsl, m15_tiny_lm.nsl
- GPU Kernels: m17_kernel_test.nsl
- Interop: m18b_interop.nsl
- Full Pipeline: gpt2.nsl

### 5.3 Error Message Audit

Spot-check critical user-facing error paths and verify they include file + line number with a human-readable message. The bar: `TypeError at main.nsl:45: expected Tensor, found String` — not a raw Cranelift panic or a message with no location. Not a full diagnostic overhaul (deferred), just verify the most common errors are usable.

---

## Scope Exclusions (Deferred)

The following are explicitly out of scope for v0.1:
- **Package manager** (`nsl pkg`) — v0.2
- **PyTorch FFI** (`to_torch`/`from_torch`/`py.call()`) — v0.2
- **Distributed training** (DDP) — v0.2
- **REPL** (`nsl repl`) — v0.2
- **LSP server** — v0.2
- **Full AST-based formatter** — v0.2 (v0.1 uses token-based)
- **Codegen error spans** — v0.2 (150+ sites need changes)
- **ROCm/Metal GPU backends** — v0.2+
- **Sparse tensor types** — v0.2+
- **Top-level let/const as globals** — architectural change, v0.2
- **`cargo install nsl-cli`** — needs `include_bytes!()` runtime embedding, v0.2

## Success Criteria

v0.1.0 is ready to ship when:
1. All examples in `examples/` compile and run (or fail with a clear error)
2. E2E smoke tests pass on all 4 CI platforms (Linux, Windows, macOS ARM, macOS Intel)
3. `nsl fmt --check` passes on all stdlib and example files
4. `nsl init myproject && cd myproject && nsl run main.nsl` works from a clean state
5. GitHub Release contains toolchain bundles for all 4 platforms
6. Library crates published to crates.io
7. CHANGELOG.md and README.md are current
8. Zero clippy warnings
