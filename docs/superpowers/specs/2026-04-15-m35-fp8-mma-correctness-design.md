# M35 FP8 MMA Correctness — Design Spec

**Date:** 2026-04-15
**Status:** Design approved, ready for implementation plan
**Scope:** Workstream A of the three-part M35 FP8 MMA maturity effort (A: correctness, B: performance, C: training integration). This spec covers **A only**.

## Context

M35 landed ~2100 LOC of FP8 infrastructure across `crates/nsl-codegen/src/fp8.rs`, `crates/nsl-runtime/src/fp8.rs`, `crates/nsl-semantic/src/fp8.rs`: `@fp8_compute` decorator extraction, sm_89+ `mma.sync` PTX for E4M3/E5M2 tensor-core matmul, Hopper `wgmma` kernel, CPU scalar fallback, running-max calibration with EMA, PerTensor / PerBlock (MXFP8) / PerChannel scaling, and E5M2 backward kernels.

Test coverage today: **41 inline unit tests**, **zero integration tests** in `crates/*/tests/`. No numerical-equivalence proof against a ground-truth reference. No end-to-end FP8 matmul correctness claim. The 39-bug audit commit `c019bac` touched FP8 as part of a broader sweep, but there's no targeted correctness test suite that would catch future regressions.

**This spec builds the correctness floor** so workstreams B (performance) and C (training integration) have a trustable base.

## Goals

1. Every FP8 op in scope has an integration test file in `crates/nsl-runtime/tests/fp8_*.rs`, independently runnable.
2. FP8 matmul (forward E4M3 + backward E5M2) agrees with a CPU scalar FP8 reference within published FP8 quantization-noise bounds.
3. MMA kernel and runtime fallback agree numerically — proves the dispatcher is algorithmically neutral.
4. Every bug surfaced during test-writing is fixed in this spec's implementation.

## Non-Goals

- **PerBlock (MXFP8), PerChannel scaling** — orthogonal scale-arithmetic variants; separate follow-up spec that reuses this spec's test harness.
- **Hopper `wgmma` kernel** — different instruction path (sm_90+); separate follow-up.
- **Calibration EMA, training-loop integration** — stateful behavior over time; lands in workstream C.
- **Performance benchmarking** — workstream B.
- **New runtime assertions** (NaN guards, scale-zero checks, format-code validation). Correctness tests catch mismatches without runtime instrumentation; if a specific assertion would catch a real bug the tests surface, it lands in its own follow-up.

## Scope

| Area | In scope | Out of scope |
|---|---|---|
| Formats | E4M3 (forward), E5M2 (backward) | — |
| Scaling | PerTensor | PerBlock, PerChannel |
| Kernel paths | `mma.sync` (sm_89+), CPU scalar fallback | `wgmma` (sm_90+) |
| Ops | matmul fwd, matmul bwd, cast, `compute_scale` | calibration EMA, `nsl_fp8_update_calibration` |

## Success Criteria

- `cargo test -p nsl-runtime` runs all new FP8 integration tests on CI (CPU-only path; MMA path feature-gated behind `cuda` + `is_sm_89_plus()`).
- Matmul output vs CPU scalar reference:
  - E4M3: max-abs relative error **≤ 2%**.
  - E5M2: max-abs relative error **≤ 10%**.
- MMA output vs fallback output: max-abs **≤ 1e-5** at all tested shapes (divergence beyond 1e-5 indicates an algorithmic difference, not f32 accumulation-ordering physics).
- Cast round-trip `f32 → FP8 → f32` within the format's max quantization step (stair-step tolerance).
- `compute_scale`: tolerance `f32::EPSILON` (1 ULP), safe against future parallel-reduction implementations.
- All-zero input to `compute_scale`: returns sentinel `1.0`.

## Architecture

The correctness surface is the runtime's FFI boundary — the public `nsl_fp8_*` functions and their scalar helpers in `compute_scale` / `quantize_fp8` / `dequantize_fp8`. Codegen is not directly exercised here: the FFI is what a compiled NSL program actually calls, and testing at that boundary catches both codegen-emission bugs (via the FFI outputs) and runtime-arithmetic bugs (via the numerical comparison).

```
┌─────────────────────────────────────────────────────────┐
│ Fixture: deterministic seed, f32 A[m,k], B[k,n]        │
│ Shapes covered: (16,16,16), (32,32,32), (64,64,64),    │
│                 (128,128,128)                           │
└───────────────────────────┬─────────────────────────────┘
                            │
                   compute_scale(A ∪ B, format)
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌─────────────────────┐              ┌──────────────────────┐
│ Reference (test):   │              │ Under test (runtime):│
│ quantize_e4m3 →    │              │ nsl_fp8_matmul(…)    │
│ dequantize →       │              │ or                   │
│ sequential FMA     │              │ fp8_matmul_e5m2_bwd  │
│ → C_ref[m,n] f32   │              │ → C_test[m,n] f32    │
└──────────┬──────────┘              └──────────┬───────────┘
           └──────────────────┬─────────────────┘
                              ▼
                   Assert: max-abs rel error
                   ≤ 2% (E4M3) / 10% (E5M2)
```

**Dispatcher comparison** (when `cuda` + sm_89+):

```
A, B → nsl_fp8_matmul (MMA)       → C_mma
A, B → fp8_matmul_cpu (fallback)  → C_fallback
Assert max-abs(C_mma - C_fallback) ≤ 1e-5
```

## Components

### New files

| Path | Purpose |
|---|---|
| `crates/nsl-runtime/tests/common/fp8_reference.rs` | Shared CPU scalar FP8 reference: quantize_e4m3, quantize_e5m2, dequantize, sequential matmul, tolerance constants (`E4M3_REL_TOL = 0.02`, `E5M2_REL_TOL = 0.10`, `DISPATCH_ABS_TOL = 1e-5`), shape fixtures. |
| `crates/nsl-runtime/tests/fp8_cast.rs` | f32 ↔ E4M3 and f32 ↔ E5M2 round-trip; stair-step tolerance per format. |
| `crates/nsl-runtime/tests/fp8_scale.rs` | `compute_scale` correctness: synthetic inputs with known max, all-zero sentinel, E4M3/E5M2 bounds. |
| `crates/nsl-runtime/tests/fp8_matmul_forward.rs` | E4M3 forward matmul vs reference, PerTensor. Shapes 16, 32, 64, 128. |
| `crates/nsl-runtime/tests/fp8_matmul_backward.rs` | E5M2 backward matmul vs reference. Same shape set. |
| `crates/nsl-runtime/tests/fp8_dispatcher.rs` | MMA path == fallback path at representative shapes (feature-gated). |

### Files modified

| Path | What changes |
|---|---|
| `crates/nsl-runtime/src/fp8.rs` | Bug fixes surfaced by the new tests only. No interface changes unless a bug necessitates one; any such change is called out in the commit. |
| `crates/nsl-codegen/src/fp8.rs` | Same — fixes only. |
| `crates/nsl-semantic/src/fp8.rs` | Same. |
| `crates/nsl-runtime/Cargo.toml` | `[[test]]` entries for the 6 new test targets, plus a `common` shared module declaration. |

### Test harness shape

`common/fp8_reference.rs` exposes:

```rust
pub const FP8E4M3_MAX: f32 = 448.0;
pub const FP8E5M2_MAX: f32 = 57344.0;
pub const E4M3_REL_TOL: f32 = 0.02;
pub const E5M2_REL_TOL: f32 = 0.10;
pub const DISPATCH_ABS_TOL: f32 = 1e-5;

pub fn quantize_e4m3(x: f32, scale: f32) -> u8;
pub fn quantize_e5m2(x: f32, scale: f32) -> u8;
pub fn dequantize_e4m3(q: u8, scale: f32) -> f32;
pub fn dequantize_e5m2(q: u8, scale: f32) -> f32;

pub struct Fp8ReferenceMatmul { pub m: usize, pub n: usize, pub k: usize,
                                 pub format: Fp8Format, pub scale: f32 }
impl Fp8ReferenceMatmul {
    /// Quantize → dequantize → sequential FMA → f32 output.
    pub fn compute_f32(&self, a: &[f32], b: &[f32]) -> Vec<f32>;
}

pub const FIXTURE_SHAPES: &[(usize, usize, usize)] =
    &[(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128)];

pub fn seeded_input(m: usize, k: usize, seed: u64) -> Vec<f32>;
```

## Data Flow

### Forward matmul test (E4M3, PerTensor)

1. Generate deterministic `f32 A[m,k]`, `B[k,n]` via `seeded_input(..., seed=0)`.
2. Compute `scale = compute_scale(A ∪ B, E4M3) = max_abs / FP8E4M3_MAX`.
3. Reference path: `Fp8ReferenceMatmul::compute_f32` — scalar quantize → dequantize → sequential FMA.
4. Under-test path: `nsl_fp8_matmul(a_ptr, b_ptr, FP8_FORMAT_E4M3, scale, ...)`.
5. Assert `max_abs(C_test - C_ref) / max_abs(C_ref) ≤ E4M3_REL_TOL` (0.02).

### Backward matmul test (E5M2)

Same flow, E5M2 format in steps 2-3, `E5M2_REL_TOL` (0.10) in step 5. Calls `fp8_matmul_e5m2_backward` (or `nsl_fp8_quantize_e5m2` + scalar matmul for the reference).

### Cast round-trip

`f32_in → nsl_fp8_cast(format) → dequantize_fp8 → f32_out`. Tolerance is the format's max quantization step at `|f32_in|`'s magnitude (non-uniform FP8 spacing — computed per value, not a single uniform ε).

### Scale compute

Synthetic `f32` array with known `max_abs`, call `nsl_fp8_compute_scale(tensor_ptr, fp8_dtype)`. Assert `(scale - expected).abs() ≤ f32::EPSILON`. All-zero input → expect `1.0` exactly.

### Dispatcher comparison (feature-gated)

Same `A, B`, run both MMA path and CPU fallback. Assert `max_abs(C_mma - C_fallback) ≤ 1e-5`. Divergence beyond 1e-5 indicates an algorithmic bug, not accumulation-ordering physics. Gated on `#[cfg(feature = "cuda")]` + runtime `is_sm_89_plus()` check; otherwise `#[ignore]` with a reason.

### Determinism

All tests seed via `rand::StdRng::seed_from_u64(0)` (or equivalent deterministic source). Running the same test twice produces bit-identical reference values. Regressions are obvious; flake is impossible.

## Error Handling & Bug Triage

When a test fails during implementation, the engineer applies this triage order:

1. **Check the reference first.** If the CPU scalar reference itself is wrong (e.g. wrong FP8 bit layout), fix the reference and re-run. The reference is the ground truth; a wrong reference produces false positives.
2. **Check tolerance bounds.** E4M3's 2% and E5M2's 10% come from published FP8 error characterizations. Near-misses at the ceiling are usually bugs in scaling or quantization, not "acceptable noise." Do not loosen a tolerance to make a failing test pass.
3. **Check format/scale path.** Most FP8 bugs look like "output close but systematically off" — typically a scale mismatch or wrong format enum reaching the kernel. Print the scale factor; compare to the reference's.
4. **Structural bugs** (wrong MMA shape, swapped operands, transpose confusion) — fix in the same commit as the surfacing test, with a comment explaining the bug and the fix. Never `#[ignore]` a failing correctness test.
5. **Scope-broadening bugs** (multi-crate refactor, dispatcher rewiring) — pause, flag, decide whether to split the spec. Do not silently balloon scope.

**Not bugs:**
- E4M3 result within 2% of reference: FP8 quantization noise, correct.
- MMA vs fallback within 1e-5 on large matrices: f32 accumulation-ordering physics, correct.
- `compute_scale` returning `1.0` for all-zero input: intentional sentinel.

**Test isolation:** each test file is an independent integration-test binary. If `fp8_matmul_forward` fails, `fp8_cast` / `fp8_scale` / others still run. No shared mutable state between files (the FP8 scale HashMap in `fp8.rs` uses per-tensor-ptr keys, so parallel tests don't collide).

## Feature Gating

- **CPU-only tests** (`fp8_cast`, `fp8_scale`, reference matmul, fallback kernel): always run on every CI configuration.
- **MMA path tests** (`fp8_dispatcher`, MMA variants in `fp8_matmul_forward`/`_backward`): `#[cfg(feature = "cuda")]` + runtime `is_sm_89_plus()` check. Test is `#[ignore]`-tagged with a reason string when the hardware floor isn't met.

## Testing Strategy

Six test binaries, each independently runnable:

| File | Tests | Runs on |
|---|---|---|
| `fp8_cast.rs` | ~4 per format × 2 formats = 8 | all configs |
| `fp8_scale.rs` | ~6 (E4M3 bounds, E5M2 bounds, all-zero, single-element, negative, large) | all configs |
| `fp8_matmul_forward.rs` | 4 shapes × {MMA, fallback} = 8 | fallback always; MMA on cuda+sm_89+ |
| `fp8_matmul_backward.rs` | 4 shapes | same |
| `fp8_dispatcher.rs` | 4 shapes | cuda+sm_89+ only |
| `common/fp8_reference.rs` | (module, no tests) | — |

All assertions use `assert!` with informative messages (`"E4M3 rel error {rel} > {tol} at shape {m}x{n}x{k}"`). On failure, the engineer can read the shape, the actual error, and the tolerance from the message.

## Commit Strategy

One commit per test file landing (with any co-discovered fix bundled in that commit). This keeps review reviewable: each commit is "integration test + fix for whatever it surfaced." Final commit if needed: any stray cleanup.

## Follow-ups (Not This Spec)

After this spec ships:
- **A2. PerBlock + PerChannel scaling correctness** — reuses this spec's harness, adds scale-arithmetic-variant tests.
- **A3. wgmma kernel correctness** — sm_90+ gate, reuses the dispatcher test's comparison structure.
- **B. Performance validation** — benchmark harness, MMA vs BF16 throughput, profile wgmma vs mma.sync.
- **C. Training integration** — calibration EMA wiring, FP8 variant of coder500m, loss-equivalence vs BF16.
