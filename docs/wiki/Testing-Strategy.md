<!-- owner: @bwiemz -->

# Testing Strategy

NSL has four layers of tests. Every PR that touches non-trivial code should add at least one.

## The pyramid

```text
      ┌──────────────┐
      │     e2e      │   real .nsl programs, full pipeline
      ├──────────────┤
      │ differential │   CPU vs GPU numerical equivalence
      ├──────────────┤
      │   snapshot   │   AST / IR / PTX stability (insta)
      ├──────────────┤
      │     unit     │   Rust functions, pure logic
      └──────────────┘
```

## Unit tests

Location: `crates/<crate>/src/**/mod.rs` inside `#[cfg(test)] mod tests { ... }` and `crates/<crate>/tests/*.rs` for integration.

Run: `cargo test -p nsl-codegen` (or any crate name).

Representative example: [`crates/nsl-codegen/src/ad_rules.rs`](../../crates/nsl-codegen/src/ad_rules.rs) — the `tests` block starting around line 731 exercises `apply_ad_rule` for every primal op (Add, Sub, Mul, Matmul, Relu, Sigmoid, Tanh, …) using hand-built `Op` structs and `matches!` assertions on the returned `AdjointExpr` variants.

### Adding one

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn my_pass_preserves_shapes() {
        let input = todo!("construct a hand-built input");
        let output = my_pass(input);
        assert_eq!(output.shape(), input.shape());
    }
}
```

## Snapshot tests — `insta`

Capture deterministic text artifacts (AST dumps, IR, PTX). Changes are reviewed by diffing the `.snap` file.

Representative example: [`crates/nsl-codegen/tests/fa_v2_snapshots.rs`](../../crates/nsl-codegen/tests/fa_v2_snapshots.rs) — each test calls a single phase emitter (e.g. `prelude::emit`, `softmax::emit`) and passes the resulting PTX string to `insta::assert_snapshot!`. Committed snapshots live in `crates/nsl-codegen/tests/snapshots/` (e.g. `fa_v2_snapshots__phase_prelude__32x32x32.snap`).

### Adding one

```rust
#[test]
fn parses_model_block() {
    let src = "model M: w: Tensor = zeros([3])";
    let ast = parse(src);
    insta::assert_debug_snapshot!(ast);
}
```

First run creates `.snap.new`. Review with `cargo insta review` and accept.

**When an existing snapshot changes in a PR**, review the diff deliberately — a snapshot change usually means either (a) your change is correct and the snapshot needs accepting, or (b) you regressed something subtle.

## Differential tests — fused vs unfused

Verifies numerical equivalence between fused and unfused code paths — runs the same `.nsl` script twice via the CLI (once with default fusion, once with `--disable-fusion`) and asserts max-abs-diff within tolerance. Catches precision regressions introduced by fusion passes.

Representative example: [`crates/nsl-cli/tests/differential.rs`](../../crates/nsl-cli/tests/differential.rs) — runs the same `.nsl` script twice via the CLI with and without `--disable-fusion`, captures stdout from both runs, and asserts max-abs-diff is within tolerance. The test gracefully skips if either run fails to compile (detected via `nsl_run()` returning `None`).

Typical tolerance: tight bounds like `1e-5` or `1e-6` (relative error on f64 stdout, both runs use same dtype and device — no CPU-vs-GPU comparison happens here).

### Adding one

- Compile the same `.nsl` function for CPU and GPU
- Run both with identical inputs
- Assert max-abs-diff below the appropriate tolerance tier

Skipped by default when no GPU is available (see GPU-gated tests below).

## GPU-gated tests

Run only when a CUDA device is present. The codebase uses two complementary mechanisms:

1. **`#[ignore]` attribute** — the test is compiled unconditionally but skipped by `cargo test` unless `--ignored` is passed. Inside the function body, a `cuda_available()` guard (`nsl_cuda_init()` + optional `NSL_SKIP_CUDA_TESTS` env var) downgrades a missing driver to a graceful skip rather than a hard failure.
2. **`#[cfg(feature = "cuda")]`** — used on a subset of tests (primarily in `nsl-codegen`) that require the `cuda` Cargo feature to be compiled at all.

Representative example: [`crates/nsl-codegen/tests/csha_cuda_launch_fused.rs`](../../crates/nsl-codegen/tests/csha_cuda_launch_fused.rs) — GPU-resident CSHA launch tests are decorated with `#[ignore]` and begin with `if !cuda_available() { return; }`. The `cuda_available()` helper calls `nsl_cuda_init()` and returns `false` if the CUDA driver is absent or if `NSL_SKIP_CUDA_TESTS` is set in the environment.

When comparing GPU kernel outputs against a reference implementation, use tiered tolerance based on the data path:

- **5e-3** — standard f32 attention outputs
- **2e-2** — FP16 attention outputs
- **4e-2** — accumulations across large head-dim (128+)

These scale with `O(sqrt(d) * ε_f16)` where `d` is the accumulation dimension and `ε_f16` is the FP16 unit roundoff.

### Running

```bash
# Run all GPU-gated tests (requires a CUDA device):
cargo test -p nsl-codegen --features cuda -- --ignored

# Run a single GPU test by name:
cargo test -p nsl-codegen --features cuda --test csha_cuda_launch_fused -- --ignored t1_forward_output_invariant

# Suppress GPU tests explicitly (e.g. in CI without a device):
NSL_SKIP_CUDA_TESTS=1 cargo test --workspace
```

### CI

The CI workflow (`.github/workflows/ci.yml`) runs on a matrix of `ubuntu-latest`, `windows-latest`, `macos-14`, and `macos-latest`. It does **not** provision a CUDA device, so GPU-gated (`#[ignore]`) tests are never triggered in CI. What CI does run:

| Step | Command |
|------|---------|
| Build | `cargo build --workspace` |
| Unit + integration tests (no GPU) | `cargo test --workspace -- --skip e2e_` |
| Lint | `cargo clippy --workspace -- -D warnings` |
| E2E smoke (Linux + Windows, blocking) | `cargo test -p nsl-cli --test e2e -- --test-threads=1` |
| E2E smoke (macOS, non-blocking) | same command with `continue-on-error: true` |

GPU tests must be run manually on a machine with a CUDA device before merging any kernel-level change.

## E2E tests — real `.nsl` programs

Location: [`crates/nsl-cli/tests/e2e.rs`](../../crates/nsl-cli/tests/e2e.rs) plus `.nsl` fixture files under `crates/nsl-cli/tests/fixtures/` and the top-level `tests/` directory.

Each test compiles and runs a `.nsl` file through the full pipeline (parse → semantic → codegen → link → execute) and compares stdout against an expected baseline. Floating-point output is normalized to six decimal places before comparison to tolerate platform-level formatting differences. E2E test failures block CI merges on Linux and Windows.

The `tests/` directory contains the full range of integration fixtures: GPU broadcast/matmul/rope shapes, source-AD training programs, checkpoint round-trips, sampling, and transformer block tests. These are the same programs exercised by the reading order in [Examples-Guide](Examples-Guide.md).

## Test discipline

- **New language feature** → unit + snapshot (AST, IR) + at least one e2e example
- **New IR pass** → unit (hand-built input) + snapshot (capture pass output) + differential (if it touches math)
- **New kernel** → unit + snapshot (PTX) + differential + GPU-gated smoke
- **Bug fix** → regression test reproducing the bug before the fix

See [Adding-a-Language-Feature](Adding-a-Language-Feature.md) for the end-to-end workflow including where tests fit.

## Common traps

- **Snapshot churn** — accepting stale snapshots without reading them. Always `cargo insta review`, never `cargo insta accept` blind.
- **Flaky GPU tests** — CUDA context leaks across tests. If a test passes in isolation but fails in a batch, suspect context. See [Runtime-Internals § GPU path](Runtime-Internals.md#gpu-path) for the `ensure_context()` rule.
- **f64/f32 tolerance** — don't use exact equality between CPU (f64) and GPU (f32) results. Use the tiered tolerance (5e-3 / 2e-2 / 4e-2) described in the Differential tests section.
- **Missing `--ignored` flag** — GPU tests silently skip (not fail) without `--ignored`. If you're not seeing a GPU test run at all, add `-- --ignored`.

---

*Last structurally verified against commit `9a1b512e` on 2026-04-21. If the crate graph or pass order in this page no longer matches reality, open an issue tagged `docs-rot`.*
