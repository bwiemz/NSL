# FASE Deferred Numerical Validation — Design

**Date:** 2026-04-14
**Status:** Design approved, ready for implementation plan
**Scope:** Item #2 of the FASE roadmap
**Depends on:** Item #1 (Deferred codegen integration, landed on `feat/fase-deferred`)

## Context

Item #1 wired FASE Deferred-mode codegen into the train-block emitter. The
smoke test proves a train block with `grad_accumulation=4` and AdamW compiles
and runs to completion, but it does not prove the emitted math is correct.
This spec adds three orthogonal tests that together validate the FASE
Deferred rewrite.

FASE-Deferred's v-update is mathematically NOT identical to standard AdamW
(CFTP §2.3 Option B): it uses `v ≈ β₂·v + (1-β₂)·mean(g²)` instead of
`v = β₂·v + (1-β₂)·mean(g)²`. These differ whenever per-micro-batch
gradients vary. "Numerical equivalence" therefore cannot mean bit-exact
for AdamW; the three-test decomposition below respects this while still
covering every load-bearing claim.

## Goals

1. **Test 1** — prove the accumulator + SGD final step are mathematically
   exact (end-to-end, through compiled code).
2. **Test 2** — prove the full AdamW Deferred pipeline matches a pure-Rust
   reference that implements the FASE-Deferred formulas (end-to-end,
   through compiled code).
3. **Test 3** — fence the intentional v-approximation against future
   "fixes" by asserting Jensen's inequality holds (pure Rust, no compiler).

## Non-Goals

- Extending `model_save` / `.nslm` to serialize optimizer state (m, v,
  m_partial).  Tests read only θ via the existing save surface.
- Peak-memory validation — that is item #5.
- GPU-vs-CPU equivalence — tests run on CPU to minimize variables.
- Additional optimizer coverage beyond SGD and AdamW — SGD+momentum, Adam
  (non-W), and the FullBuffer fallback are already covered by item #1's
  snapshot tests.

## Design Decisions

### D1. Pure-Rust references, not a codegen-path toggle

Test 1 and Test 2 each implement the reference optimizer trajectory in
~20–40 lines of Rust and compare the compiled program's θ against that
reference. The reference shares no code with the codegen under test, so a
shared bug (wrong β₂ default, transposed fixture weights, incorrect
gradient scaling) cannot mask a real failure.

Rejected alternative — toggling the planner to return `FullBuffer` and
comparing two compiled outputs — because both paths share locals, helpers,
and parsing. A shared bug would pass. A pure-Rust reference is auditable
in isolation.

### D2. θ-only visibility (no changes to save surface)

`model_save(m, path)` serializes the model's parameter fields only, not
optimizer state. Rather than extending the save format, the tests are
reformulated so every claim is observable through θ after N micro-batches:

- Test 1 (SGD exact): θ after one window = `θ_init - lr · mean(g_1..g_N)`.
  Rust reference computes this directly.
- Test 2 (AdamW pipeline): θ after W windows fully determines whether the
  FASE-Deferred recipe was emitted correctly, given identical gradients
  and hyperparameters.  Running three windows exercises persistent m/v
  state across windows.
- Test 3 (Jensen fence): pure Rust, compares the FASE-Deferred v-update
  against the standard AdamW v-update; no compiler involvement, no save
  format needed.

Localization trade-off: if Test 2 fails, the failure surface is the
AdamW-specific recipe ops (`ScalarMulAdd`, `SquaredAccumulate`,
`SqrtPlusEps`, `Div`, `Update` — plus the Tmp register scratch handling).
Test 1 passing + Test 2 failing isolates to that surface.  This is
acceptable given it is a small, well-bounded set of call sites.

### D3. Deterministic fixtures — zero RNG, zero data loader

Both `.nsl` fixtures use:

- A single linear layer: `w: Tensor<[2, 1], f32>`.
- Compile-time constant init weights (e.g. `w = [[0.5], [-0.3]]`).
- Four hard-coded input batches baked into the NSL source as constant
  arrays (no `dataset` / `MemoryMapped` / tokenizer).
- Fixed target values per batch.
- MSE loss: `(y - target)²` scalar.

With MSE-on-linear-layer the gradients are closed-form:
`dL/dw = 2·(x·w - target)·xᵀ`.  The Rust reference computes this
directly in f32, eliminating any uncertainty about what "the gradient"
is.  No autograd, no RNG, no I/O.

## Architecture

### Test 1 — SGD exact equivalence (integration test)

- **Fixture:** `crates/nsl-codegen/tests/fixtures/fase_deferred_sgd_equivalence.nsl`
- **Optimizer:** plain SGD, `lr = 0.01`.
- **Windows:** 1 (4 micro-batches, one optimizer step).
- **End of fixture:** `model_save(m, <temp>/sgd_out.nslm)`.
- **Rust reference (`sgd_reference`):** compute the 4 gradients closed-form,
  take their mean, apply `θ -= lr · mean(g)` once.
- **Assertion:** `|θ_compiled[i] - θ_reference[i]| / max(1.0, |θ_reference[i]|) < 1e-6`
  for every element.
- **What it proves:** pre-scaled accumulation `m_partial += g/N`, the SGD
  final-step dispatch in `stmt.rs`, the per-parameter loop emission, and
  the buffer ownership contract in `fase_emit_final_step` (borrowed pointers,
  caller-managed θ) are all correct.

### Test 2 — AdamW Deferred pipeline (integration test)

- **Fixture:** `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_equivalence.nsl`
- **Optimizer:** `AdamW(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)`.
  All hyperparameters explicit (no defaults) to avoid drift against the Rust
  reference.
- **Windows:** 3 (12 micro-batches; persistent m/v across windows).
- **End of fixture:** `model_save(m, <temp>/adamw_out.nslm)`.
- **Rust reference (`adamw_fase_deferred_reference`):** faithful implementation
  of the FASE-Deferred math, f32 arithmetic throughout:

  ```
  for window in 0..3:
      m_partial = mean(g_{4·window..4·(window+1)})
      m  = β₁·m + (1-β₁)·m_partial
      v  = β₂·v + (1-β₂)·m_partial²
      step += 1
      bc1 = 1 - β₁^step
      bc2 = 1 - β₂^step
      m_hat = m / bc1
      v_hat = v / bc2
      θ -= lr · (m_hat / (√v_hat + ε) + wd·θ)
  ```

- **Assertion:** `|θ_compiled[i] - θ_reference[i]| / max(1.0, |θ_reference[i]|) < 1e-5`
  after 3 windows.
- **What it proves:** the AdamW recipe ops lower correctly; the Tmp
  register is allocated, threaded across `SqrtPlusEps → Div`, and freed;
  `m_partial` is zeroed between windows by `fase_emit_final_step`; bias
  correction runs with the right step count.

### Test 3 — Jensen-inequality fence (pure Rust unit test)

- **Location:** `crates/nsl-codegen/src/fase_optimizer.rs`, in the
  existing `#[cfg(test)] mod tests` block.
- **Body:** two local functions — `fase_deferred_v_update(v, m_partial)`
  and `standard_adamw_v_update(v, gradients)` — that each perform one
  v-update step with `β₂ = 0.999`.  Call them with a synthetic
  non-constant gradient sequence (e.g. `[1.0, 2.0, 0.5, 1.5]`).
- **Assertions:**
  1. `v_fase >= v_standard` element-wise (Jensen: `mean(g²) ≥ mean(g)²`).
  2. `v_fase - v_standard > 0` strictly (enforces non-constant input was
     used).
- **What it proves:** the FASE paper's approximation is intentional and
  in the documented direction.  Any future edit that accidentally
  implements the standard formula fails this test immediately.
- **Cost:** runs in microseconds; zero codegen coupling; can never flake.

## Components

### `crates/nsl-codegen/tests/common/nslm_reader.rs` (new, ~40 LOC)

Minimal `.nslm` parser.  Accepts a file path, returns
`HashMap<String, Vec<f32>>` keyed by tensor name.  Format (already
defined by `crates/nsl-runtime/src/checkpoint.rs`):

- 4 bytes magic (`NSLM`)
- 4 bytes version (u32, little-endian)
- 8 bytes header size (u64, little-endian)
- `header_size` bytes of JSON: `{"params":[{"name":..., "shape":..., "dtype":..., "offset":..., "nbytes":...}, ...]}`
- Aligned (64-byte) data payload; each tensor at its declared `offset`
  for `nbytes` bytes.

Only supports `dtype=f32` and `dtype=f64` (cast f64→f32 during read).
Reader does NOT support multi-dim tensors natively — it flattens; tests
compare element-by-element after flattening.

### `crates/nsl-codegen/tests/fase_numerical_validation.rs` (new, ~150 LOC)

Three `#[test]` functions:

- `sgd_exact_equivalence` — Test 1.
- `adamw_fase_deferred_pipeline_equivalence` — Test 2.
- `jensen_fence_lives_in_fase_optimizer` — NOT here; this is a unit test
  inside `fase_optimizer.rs` and not duplicated in this integration file.

Each integration test:

1. Creates a tempdir.
2. Spawns `CARGO_BIN_EXE_nsl run <fixture> --device=cpu` (or whatever flag
   forces CPU; discover during implementation).
3. Reads `<tempdir>/<out>.nslm` via the reader.
4. Computes the Rust reference.
5. Asserts element-wise tolerance.

Tempdir via `tempfile::TempDir` if the crate is already a workspace dep;
otherwise a `std::env::temp_dir()` subdir named after the test, cleaned on
`Drop` via a small RAII guard.

### `crates/nsl-codegen/tests/fixtures/fase_deferred_sgd_equivalence.nsl` (new)

Minimal NSL source: 2-element linear layer, 4 hard-coded input batches,
fixed targets, MSE loss, `train(grad_accumulation=4, epochs=1)` with
`optimizer: Sgd(lr=0.01)`, then `model_save(m, env("NSL_TEST_OUT"))`.

Uses `env()` (or equivalent) to read the output path from the
environment — the Rust harness sets `NSL_TEST_OUT` before spawning the
subprocess.  If no `env()` builtin exists, use a fixed path under
`./target/tmp/` per test — implementation plan discovers which is
available.

### `crates/nsl-codegen/tests/fixtures/fase_deferred_adamw_equivalence.nsl` (new)

Same shape as the SGD fixture but three windows (`epochs=1, grad_accumulation=4`
with 12 batches, OR `epochs=3, grad_accumulation=4` with 4 batches — picker's
choice based on what the train-block surface supports cleanly) and an AdamW
optimizer with all hyperparameters explicit.

## Data Flow

```
[fixture.nsl]
   │  (nsl run --device=cpu)
   ▼
[final.nslm] ──────────────┐
                           │
                           ▼
                     [nslm_reader]
                           │
                           ▼
                   HashMap<name, Vec<f32>>
                           │
                           ▼
                     [assert_close]
                           ▲
                           │
                    [rust_reference]
                           ▲
                           │
          (same gradients, same optimizer math, f32)
```

## Tolerances

- Test 1 (SGD, 1 window): `1e-6` relative.
- Test 2 (AdamW, 3 windows): `1e-5` relative.  Wider because AdamW's
  sqrt/div chain amplifies rounding, and three windows accumulate error.
- Test 3 (Jensen): strict `>` comparison — pure f64 math, no tolerance.

If any tolerance turns out to be too tight in practice, revisit in the
implementation plan rather than loosening here.

## Risks

1. **`nsl run` on CPU.** `nsl check` alone isn't enough — we need to
   execute.  Runtime setup may require a device flag or a runtime
   precondition (CUDA context) that the smoke test did not exercise.
   Investigate during implementation; if it turns out GPU is required
   on this machine, mark the tests `#[cfg_attr(not(gpu), ignore)]`
   (or equivalent) and document.
2. **Fixture surface-syntax drift.** NSL's train-block surface for
   `Sgd(lr=0.01)` etc. must be verified against existing examples
   before writing the fixture.  If `Sgd` isn't the exact identifier, use
   whatever is.  Implementation plan does the grep.
3. **`.nslm` layout of `f32` tensors.** The checkpoint.rs writer stores
   the raw bytes and records `dtype`/`nbytes`; the reader must honor
   both. Writing a round-trip test against an existing coder-50M or
   m14 fixture first would catch layout surprises.
4. **Bias-correction step counter.** The Rust reference uses a local
   `step` counter incremented per window.  If the NSL AdamW implementation
   uses per-micro-batch step counting (instead of per-optimizer-step),
   the references diverge — verify via a quick read of the AdamW stdlib
   before baking the reference.

## Success Criteria

- `cargo test -p nsl-codegen --test fase_numerical_validation` passes on
  CPU.
- `cargo test -p nsl-codegen --lib fase_optimizer::tests::jensen_fence_lives_in_fase_optimizer`
  passes.
- No changes to runtime, codegen, planner, or save surface.  Only new
  test code and two new fixtures.

## Follow-Ups

This spec closes item #2 of the FASE roadmap.  Remaining items:

- **Item #1b** — per-parameter interleaving inside backward (peak memory win).
- **Item #3** — two-phase gradient clipping codegen; removes the
  temporary `grad_clip → FullBuffer` downgrade landed in item #1.
- **Item #4** — M36 memory-planner wiring.
- **Item #5** — peak-memory regression test.
- **Item #6** — `nsl check --training-report` CLI.
