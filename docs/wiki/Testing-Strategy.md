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

### The certification tiers

`scripts/gpu-tier.sh` is the front door for running GPU certification at a
chosen depth. Each tier answers a different question:

```bash
scripts/gpu-tier.sh smoke       # "is the GPU + harness sane right now?"
scripts/gpu-tier.sh certify     # "does everything the tree certifies hold?"
scripts/gpu-tier.sh endurance   # "does production-scale 1B training complete?"
```

`smoke` runs the curated canary (`tools/gpu-canary.txt` via `tools/gpu-test.sh`,
one process per test, every entry a real numerical claim against a CPU
reference). `certify` runs the full certification lane below. `endurance` runs
`models/benchmarks/endurance_1b.py` — 1B parameters at sequence length 2048:
the SR-BF16 endurance arm, the f32 reference arm, and checkpoint-resume in a
fresh process. Measured budgets live in the script header and in
`models/benchmarks/GPU_TIERS_2026_08_24.md`.

Every tier **refuses to start when the device is busy** or when another
guarded run holds the machine-wide lock — `scripts/gpu-guard.sh`, which also
runs the workload in its own process group so killing a tier cannot orphan a
child still holding VRAM. The refusal is a hard exit, not a warning: a sweep
that warned and proceeded on a busy device lost two arms of an LR sweep to an
orphaned run's resident 22 GB (2026-08-19). The same guard is called by every
campaign driver under `models/benchmarks/`.

Note the axis distinction: `gpu-tier.sh` selects depth/duration, while
`gpu-cert.sh --tier` (below) selects gate *capability classes* within the
certify tier. `gpu-tier.sh certify --tier all` composes the two.

### The certification lane (the `certify` tier)

`scripts/gpu-cert.sh` is the full sweep: it discovers every `#[ignore]` test in
the tree, classifies each one, and runs the classes that need a device.

```bash
scripts/gpu-cert.sh --list             # inventory as TSV, no build
scripts/gpu-cert.sh --run --tier gpu   # the device-requiring gates
scripts/gpu-cert.sh --check-inventory  # drift gate (GPU-free; runs in CI)
scripts/gpu-cert.sh --check-long-arms  # timeout-override gate (GPU-free; CI)
```

Classification is what makes the sweep safe to automate. Four `#[ignore]`
tests are fixture and baseline *generators* — running them under `--ignored`
would silently rewrite the reference data other gates compare against — and
each is denied by an explicit rule on its reason string, file suffix, or
function name. Diagnostics that assert nothing, gates blocked on unlanded
work, and tests needing opt-in cargo features are likewise excluded, as are
`cpu-stub` placeholders — tests under `#[cfg(not(feature = "cuda"))]` exist
only in non-cuda builds, so the cuda-featured binaries the lane runs compile
them out and any RUN classification would report a permanent NOTFOUND. Anything
the ruleset does not recognise is classified `unclassified` and never run; it
appears in `--list` and in `ci/gpu-cert-manifest.tsv`, so the drift gate still
tracks it even though it is absent from the run report.

The lane **refuses to start** if `NSL_SKIP_CUDA_TESTS` is set, if
`CUDA_VISIBLE_DEVICES` is empty, or if `nvidia-smi` reports no device. Most
GPU tests early-return as a *pass* when no device is available, so a sweep
under those conditions would report green having executed nothing — worse than
not running at all.

`--tier gpu` is the default and covers the `gpu` and `gpu-inferred` classes —
397 gates as of this writing. It does **not** include the `toolchain` (13),
`multiproc` (6), or `isolate` (1) classes; run those separately. These counts
are not mechanically gated, so treat them as indicative; `scripts/gpu-cert.sh
--list | cut -f3 | sort | uniq -c` is the authority.

Every gate gets `NSL_CERT_TIMEOUT` seconds (default 1200), and a target's batched run gets that times its gate count, clamped to `NSL_CERT_BATCH_TIMEOUT` (default 3600). The
handful of gates that drive a full training run to convergence need longer, and
when the budget is too short the gate is reported `TIMEOUT` — which reads
identically to a hung kernel. `ci/gpu-cert-long-arms.txt` raises the budget for
named targets, as `max(NSL_CERT_TIMEOUT, entry)` so a deliberately raised
global is never shortened back. It is a separate file rather than a manifest
column because `--write-manifest` regenerates the manifest verbatim from a
four-field scanner and `--check-inventory` diffs it byte-exactly; a fifth
column would go permanently red and then be erased. `--check-long-arms`
validates the format and cross-checks every path against the manifest, so a
test-file rename fails CI instead of silently reverting the gate to 900s.

A red gate's full output is written to `logs/` beside the report, along with
`run-metadata.tsv` (tier, features, and the timeout actually in force) — the
nightly workflow uploads all of it, plus the manifest, known-red list and
long-arms table, so an artifact stays interpretable without checking out the
commit it was produced from.

Set `TMPDIR` to a disk-backed path before a full run. Many gates spawn `nsl`
builds, and on a machine where `/tmp` is tmpfs the sweep can exhaust it — the
resulting linker errors surface as generic "nsl run failed" panics that read
like compiler bugs.

Pre-existing failures are recorded in `ci/gpu-cert-known-red.txt` with a note
each; only *new* red fails the lane. When a target goes red the lane re-runs
each of its gates in a separate process, because a faulting kernel poisons the
CUDA context for every later test sharing that process — without this, one
real bug reports as a cluster.

### Canary (the `smoke` tier)

For a quick "is the GPU path working at all" check there is a curated canary
(`tools/gpu-canary.txt`) — every entry makes a real numerical claim against a
CPU reference and has been observed green on the reference GPU:

```bash
tools/gpu-test.sh                # run the canary, one process per test
tools/gpu-test.sh --list         # show the manifest
tools/gpu-test.sh --filter csha  # only matching entries
```

(`tools/gpu-test.ps1 -Canary` is the PowerShell equivalent for Windows.)
This is an acceptance check for the harness itself, not coverage — use
`scripts/gpu-cert.sh` for coverage.

See [GPU-Test-Harness](GPU-Test-Harness.md) for the full reference, the canary
set, and the "structural pass ≠ numerical pass" known-blocked list.

### Structured runtime events (`NSL_EVENTS`)

Setting `NSL_EVENTS=<path>` makes the runtime append one JSON object per
line — `{"v":1,"seq":N,"kind":"...","step":S|null,"fields":{...}}` — for
every counter reporter (`[zero]`, `[weight-stream]`, `[csla]`,
`[fase-fused]`, `[wgrad-accum]`, the launch counters, `[grad-integrity]`
when armed) and for `[gpu-mem]` at every step boundary with **exact bytes**
(the stderr line rounds to MB and throttles after step 5; events do
neither). Prefer reading events by field name over regexing stderr: the
stderr lines are unchanged and stay gated by their own env vars, but they
carry append-only/positional hazards the events don't have.

The schema registry is `exec_markers::EVENT_SCHEMAS` (same file as the
marker registry); `events_stream_gate.rs` pins the envelope, required
fields, stderr/event value agreement, byte-identical marker lines with
events on or off, and that an unwritable path warns once without failing
the run. The compiler-side decision prose (`[ccr]`, `[muon]`,
`[lm-head-fusion]` etc. from `stmt.rs`) is deliberately NOT in the event
stream — it is free-text rationale from lowering code with no shared
structure to hoist, and wants a `pass_trace`-style compile-report collector
instead; that is recorded as follow-on work, not silently skipped.

### CI

The CI workflow (`.github/workflows/ci.yml`) runs on a matrix of `ubuntu-latest`, `windows-latest`, `macos-14`, and `macos-latest`. It does **not** provision a CUDA device, so GPU-gated (`#[ignore]`) tests are never triggered in CI. What CI does run:

| Step | Command |
|------|---------|
| Build | `cargo build --workspace` |
| Unit + integration tests (no GPU) | `cargo test --workspace --no-fail-fast -- --skip e2e_` |
| Lint | `cargo clippy --workspace -- -D warnings` |
| E2E smoke (Linux + Windows, blocking) | `cargo test -p nsl-cli --test e2e -- --test-threads=1` |
| E2E smoke (macOS, non-blocking) | same command with `continue-on-error: true` |

The unit-test step carries `--no-fail-fast` deliberately. Without it `cargo
test` stops at the first failing test *binary* and never builds or runs the
rest, so a red run names exactly one broken target however many are broken —
PR #455 hit four independent `windows-latest` failures and paid four serial
~25-minute round-trips to see them, one per run. A green run is unaffected
(nothing aborts early, so there is nothing to skip) and any failing target
still fails the job. The cost is paid only by red runs, which now run to
completion instead of aborting — one longer red run in exchange for not
rediscovering the next failure a round-trip later.

CI additionally runs two build-free agreement gates: `doc-agreement`
(`scripts/check-doc-agreement.sh`) and `gpu-gate-inventory`
(`scripts/gpu-cert.sh --check-inventory`). The latter cannot execute GPU
tests, but it proves the gate manifest in `ci/gpu-cert-manifest.tsv` still
matches the source — so a gate cannot be renamed, deleted, or silently
un-`#[ignore]`d without that showing up in the same commit. It carries its own
anti-vacuity step that removes a gate and requires the check to go red.

GPU tests must be run manually on a machine with a CUDA device before merging
any kernel-level change: `scripts/gpu-cert.sh --run --tier gpu`.

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
