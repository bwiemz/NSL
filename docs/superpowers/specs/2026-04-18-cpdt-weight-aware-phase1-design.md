# CPDT Weight-Aware Refinement — Phase 1 Design

**Date:** 2026-04-18
**Status:** Design (brainstormed, awaiting user file-level review)
**Target branch:** `feat/cpdt-weight-aware-phase1`
**Predecessor:** [CPDT Pipeline Integration](2026-04-15-cpdt-pipeline-integration-design.md) (PR #37, merged 2026-04-16)
**Successor stub:** §12 (Phase 2 — spectral analysis + sidecar cache)
**Research source:** [CPDT Research.pdf](../../research/CPDT%20Research.pdf) (Brandon Wiemer, April 2026)

---

## 1. Goal, Non-Goals, Pipeline Position, Module Layout

### 1.1 Goal

CPDT's Phase 1 weight-aware refinement. When pre-trained weights are available, CPDT computes a per-parameter sensitivity score and assigns precision tiers (FP32 / FP16 / INT8 for Adam's `m` and `v` state) more accurately than the pre-weight-aware heuristic alone. The unified sensitivity scorer uses three cheap factors:

- `position_criticality(l, L)` — categorical by layer position (first/last/near-extreme/middle)
- `element_count(W)` — from `AppliedLayer` shape metadata
- `gradient_magnitude_est(W) = ||W||_F / sqrt(numel(W))` — RMS magnitude when weights present; calibrated constant `CALIB_K` when absent

One code path, not two. No-weights output is calibrated to match the pre-refactor heuristic byte-identical on a baseline corpus; weights-present output can diverge based on real tensor magnitudes. This eliminates the two-plans-for-one-model drift failure mode.

### 1.2 Non-Goals (explicitly deferred, each with a tracked follow-up milestone)

1. **Spectral condition analysis** — deferred to Phase 2 (§12). Phase 2 adds spectral as a fourth multiplicative factor in the same scorer, plus the `.cpdt-sensitivity.json` sidecar cache modeled on `wggo_weight_analysis_cache.rs`.
2. **MoE support (AST detection + router-affinity pruning + per-expert tier-assignment coverage)** — deferred as a single unified "MoE-CPDT slice." The three parts ship together; none land in isolation. The dependency is structural (router-affinity requires MoE detection; per-expert tier assignment requires router-affinity).
3. **IR mutation / communication-op emission / fused quantized optimizer codegen** — separate CPDT slices. Phase 1 only refines the *plan*; it does not materialize the plan into IR.
4. **Centralized `WeightMap::load` shape validation** — backlog architectural refactor. Phase 1 does its own inline validation in `cpdt::run` following the existing per-consumer pattern (ZK, WGGO, weight-analysis each validate independently today). A centralized pass is a separate "consolidate weight validation" refactor milestone.
5. **Post-scoring normalization** (`@cpdt(score_normalization=...)`) — added later only if either (a) weighted disagreement on a shipped NSL model's weight file exceeds 5% against an extended baseline corpus (measurable via the §8 passive diagnostic), or (b) a user-filed issue demonstrates a workflow requiring cross-checkpoint tier distribution comparison. First clause is a concrete measurable trigger.
6. **`nsl cpdt calibrate` subcommand** — Phase 1 ships `tools/cpdt_calibrate.rs` as a dev-only binary (reproducibility artifact); a user-facing subcommand is not in scope. The `CPDT_CALIB_K` env var is the supported escape hatch.
7. **Coverage for conv layers, sparse / low-rank structured weights** — listed as known-uncovered shape regimes. When these become NSL targets, calibration corpus extension is needed.

### 1.3 Pipeline Position

Unchanged from today: `compile_quant_block` runs **Calibration → WGGO → CSHA → WRGA → CPDT**.

Within `cpdt::run`, the order becomes:

1. Validate `WeightMap` (if present) against WGGO-produced `AppliedPlan` — shape, dtype, tensor-name cross-check.
2. Build `SensitivityScorer` from `AppliedPlan` + optional `WeightMap`.
3. Score every parameter; assign tier via `assign_tier(score, layer_kind)`.
4. Existing ZeRO / comm / joint-solver passes continue as before.

**Data-flow diagram** (made explicit per the §1 addendum — WGGO is visibly its own stage):

```
CLI
 └─► WeightMap::load (from --weights or AST load_safetensors(...))
      │
      │
      ▼
[WGGO: apply(&InterLayerPlan, &[LayerIlpSolution]) → AppliedPlan]
      │
      │
      ▼
cpdt::run(CpdtInput { weights, applied, ... })
      │
      ▼
[cpdt_sensitivity::validate(&weights, &applied)] — shape/dtype/name
      │
      ▼
[SensitivityScorer::new(applied, weights) → PrecisionPlan]
      │
      ▼
[existing solver chain: search_zero, plan_experts, joint::solve]
```

### 1.4 Module Layout

- **NEW:** `crates/nsl-codegen/src/cpdt_sensitivity.rs` — unified `SensitivityScorer` (concrete type, not a trait), per-parameter score computation, tier-boundary policy, `ANALYSIS_VERSION` constant. Consumes `&WeightMap` read-only. Does not extend `weight_aware.rs`, does not add methods to `WeightMap`, does not share derived computations with `SparsityInfo` / constant-folder / dead-weight-eliminator. If the scorer needs a weight-access pattern `WeightMap` doesn't expose, the fix is adding a read-only accessor on `WeightMap`, not duplicating state.
- **NEW:** `crates/nsl-codegen/src/cpdt_tier_apply.rs` — tier-*application* utilities (tier → IR emission helpers) migrated from the deleted `cpdt_precision.rs`. Filename honestly describes contents: this module applies tiers; `cpdt_sensitivity.rs` decides them.
- **DELETED:** `crates/nsl-codegen/src/cpdt_precision.rs` — all tier-*assignment* code migrates into `cpdt_sensitivity.rs`; tier-*application* code migrates into `cpdt_tier_apply.rs`. No `_legacy` shim. The file ceases to exist after the refactor commit.
- **NEW (Phase 1 spec commitment, Phase 2 implementation):** sidecar cache architecture documented in §12. Phase 1 ships the `ANALYSIS_VERSION` constant and the commit-rule doc-comment; Phase 2 ships the cache reader/writer.
- **No changes:** `cpdt_zero.rs`, `cpdt_comm.rs`, `cpdt_joint.rs`, `cpdt_expert.rs`, `cpdt_optim.rs`, `weight_aware.rs`, `wggo_apply.rs`. One exception: a one-line `// NOTE:` comment added to `cpdt_joint.rs::solve` marking the tier-derived-aggregate reading contract (§8.3).

---

## 2. CLI Surface & Weight Ingestion

### 2.1 Four-Case Decision Table

The existing global `--weights <path>` flag (already consumed by `--weight-analysis`, `--zk-weights`, `--wggo-weights`, and WRGA standalone builds) is CPDT's weight-ingestion surface. No new CLI flag. This makes the single-file-across-consumers consistency guarantee structural rather than convention-based: all compile-time weight consumers read the same file; if the file is wrong, everything fails together visibly.

**Auto-detection surface:** AST walk for `load_safetensors("<path>")` calls at module scope or inside model constructors. The existing `load_safetensors` intrinsic in `crates/nsl-semantic/src/builtins.rs` and `crates/nsl-codegen/src/expr/calls.rs` is the reference — detection reuses the same name resolution.

**Decision table:**

| AST `load_safetensors(...)` ref | `--weights` flag | Behavior |
|---|---|---|
| Present | Absent | Auto-detect; plan uses AST-referenced path |
| Present | Present | Flag overrides AST path; stderr warning (see §8.2) |
| Absent | Present | Use flag path |
| Absent | Absent | Error when `--cpdt` is active; three-option resolution message (see §2.3) |

### 2.2 Plan-Time Validation Gate

Whichever source resolved the weight file path (flag or AST), `cpdt::run` validates the loaded `WeightMap` against WGGO's `AppliedPlan` before any sensitivity scoring fires. The validation is a pass over `AppliedPlan.layers` cross-checking:

- **Shape mismatch:** `AppliedLayer.shape != WeightEntry.shape` for any tensor referenced by an `AppliedLayer` → fast error, surfaces the layer name, expected shape, actual shape.
- **Dtype mismatch:** `AppliedLayer.dtype != WeightEntry.dtype` → fast error, surfaces the dtype mismatch.
- **Missing tensor:** `AppliedLayer` references a tensor name absent from `WeightMap` → fast error, surfaces the missing name.

Failure produces a plan-time error, not a runtime error. The compiler exits non-zero with a clear diagnostic line.

### 2.3 Error Messages

**Case 4 (absent both) error** — load-bearing for user onboarding; spell the resolution rather than terse:

```
error: CPDT weight-aware tier assignment requires weights. Either:
  1. Add `m.load_safetensors("path/to/weights.safetensors")` to your NSL source
  2. Pass `--weights <path>` to nsl build
  3. Disable weight-aware CPDT with `@cpdt(weight_aware=false)` to fall back
     to heuristic-only tier assignment (produces same tiers as pre-weight-aware
     behavior)
```

**File-not-found error** (e.g., typo in `--weights` path, file in different working directory, permissions) — must show the specific path, not a vague "failed to load weights":

```
error: --weights <path> could not be read.
  path: /full/resolved/path/to/the/attempted/weights.safetensors
  cause: <underlying I/O or parse error from WeightMap::load>
```

Same behavior applies to AST `load_safetensors("...")` failures: existing `WeightMap::load` error propagation must produce a clear message showing the attempted path.

### 2.4 Format-Override Principle (Future-Loader Stub)

If multiple weight-loader formats are added in the future (e.g., `.gguf`, `.pth`, `.nslm`), the `--weights` flag's path extension determines loader selection; the AST's `load_<format>(...)` call is the default when `--weights` is absent. The two sources are never combined — the flag's format wins entirely when present. This rule matters only hypothetically today (only `.safetensors` is supported), but documenting it now prevents invention of inconsistent rules when new loaders land.

---

## 3. Sensitivity Scorer & Tier Assignment

### 3.1 Formula

```
sensitivity(W, l, kind) = gradient_magnitude_est(W) × position_criticality(l, L) / element_count(W)
```

Three factors, each with a specific concrete form. No clamping, no normalization, no range constraint on any factor's output — the multiplicative structure of the formula means each factor's physical units compose, and pre-constraining one factor breaks that composability (important for Phase 2 spectral integration).

### 3.2 `gradient_magnitude_est(W)`

```rust
fn gradient_magnitude_est(entry: Option<&WeightEntry>) -> f64 {
    match entry {
        None => CALIB_K,                                     // no-weights path
        Some(w) => {
            let sum_sq: f64 = w.iter_f32().map(|x| (x as f64).powi(2)).sum();
            (sum_sq / (w.numel() as f64)).sqrt()              // ||W||_F / sqrt(numel)
        }
    }
}
```

- No-weights path: returns the calibrated constant `CALIB_K` (see §4). This is the only weight-dependent factor; absence of weights collapses the formula to a function of `position_criticality` and `element_count` alone.
- Weights-present path: single pass over tensor bytes, arithmetic at f64 precision (converts from f16/bf16/f32 storage as needed). Physical units: same as the weight values themselves (typically in the `[0.01, 1.0]` regime for trained transformer weights).

### 3.3 `position_criticality(l, L, α)`

Piecewise over layer index `l` in a model of depth `L`, parameterized by calibration tunable `α ∈ [0, 1]`. The `L >= 4` guard makes the small-`L` degeneracy explicit at the code site rather than emergent from branch ordering:

```rust
fn position_criticality(l: usize, L: usize, alpha: f64) -> f64 {
    debug_assert!(L >= 1 && l < L);
    if l == 0 || l == L - 1 {
        return 2.0;  // first/last layer
    }
    if L >= 4 && (l == 1 || l == L - 2) {
        return 1.0 + alpha;  // near-extreme; only defined for L >= 4
    }
    1.0  // middle layer, or any non-extreme position when L < 4
}
```

**Small-`L` behavior documented explicitly:** For `L < 4`, the near-extreme regime is definitionally empty — there is no layer that is neither first/last nor middle. Calibration of `α` therefore uses only fixtures with `L ≥ 4`; `calib_tiny` (L=2) exercises the first/last and middle branches but not the `α`-tuned branch. This is intentional, not a coverage gap.

### 3.4 `element_count(W)`

Reads `AppliedLayer.numel()` directly — no computation, no weight-file access. Already present on `AppliedLayer` as of WGGO PR #34.

### 3.5 Tier Assignment

Four tiers, mapping score ranges to Adam `m` / `v` precision pairs:

| Tier | Variant | Precision (m, v) | Bytes/param | Assigned when |
|---|---|---|---|---|
| 0 (High) | `Tier::High` | FP32, FP32 | 8 | `sensitivity > T0` |
| 1 (Medium) | `Tier::Medium` | FP16, FP32 | 6 | `T1 < sensitivity ≤ T0` |
| 2 (Low) | `Tier::Low` | INT8, FP16 | 3 | `T2 < sensitivity ≤ T1` |
| 3 (Very low) | `Tier::VeryLow` | INT8, INT8 | 2 | `sensitivity ≤ T2` |

`T0 > T1 > T2 > 0`. Exact values produced by calibration (§4).

### 3.6 Layer-Kind Override

The research is prescriptive about norm and embedding layers ("norms, embeddings, first/last layer → FP32"). The formula alone cannot cleanly produce this behavior because the `element_count` divisor for a large embedding (e.g., 49152 × 1280) drives sensitivity very low regardless of magnitude.

**Resolution:** hard layer-kind override, applied *after* the formula score is computed, inside `assign_tier(score, layer_kind)`:

```rust
fn assign_tier(score: f64, layer_kind: LayerKind) -> Tier {
    match layer_kind {
        LayerKind::Norm | LayerKind::Embedding => return Tier::High,
        _ => {}
    }
    // Formula-score path for all other layer kinds.
    if score > T0 { Tier::High }
    else if score > T1 { Tier::Medium }
    else if score > T2 { Tier::Low }
    else { Tier::VeryLow }
}
```

The formula-computed score is still stored in the plan (for future debugging, spectral integration, cache keying). The *tier label* is `Tier::High` for override kinds regardless of score.

**Why a hard override rather than a multiplicative boost factor:** the research treats layer-kind as a categorical prior, not a continuous signal. A boost factor (e.g., `layer_type_boost(Norm) = 100.0`) produces the same tiers as the override in practice while making the tier decision harder to read at a glance. Hard override is simpler, more faithful to the research framing, and doesn't hide the categorical rule behind continuous arithmetic.

### 3.7 `ANALYSIS_VERSION`

```rust
pub const ANALYSIS_VERSION: u32 = 1;
```

**Rule** (documented in module doc-comment and referenced in §12 Phase 2 stub):

> Any change to the sensitivity formula, factor computation (`gradient_magnitude_est`, `position_criticality`, `element_count`), or tier-boundary policy (`assign_tier`) MUST bump `ANALYSIS_VERSION` in the same commit. Phase 2's sidecar-cache key includes this field; caches from older versions are ignored automatically. Phase 2 adds a CI check (grep-based) that flags diffs touching `fn compute_score`, `fn assign_tier`, `fn gradient_magnitude_est`, or `fn position_criticality` without a matching `ANALYSIS_VERSION` bump.

Phase 1 ships the constant and the rule. Phase 2 ships the CI enforcement. Writing the rule in Phase 1's spec means Phase 2 inherits a committed discipline rather than inventing one.

---

## 4. Calibration

Phase 1 ships five calibrated constants. Their values produce unified-scorer output that matches the pre-refactor heuristic byte-identical on the no-weights path, on the baseline corpus of synthetic fixtures (§5).

### 4.1 Constants

```rust
// All constants derived against tests/fixtures/cpdt_calibration/baseline_heuristic.json.
// Any edit to the formula or these thresholds MUST bump ANALYSIS_VERSION.

pub const CALIB_K: f64 = /* tuned */;     // neutral value of gradient_magnitude_est when weights absent
pub const CALIB_T0: f64 = /* tuned */;    // High ↔ Medium threshold
pub const CALIB_T1: f64 = /* tuned */;    // Medium ↔ Low threshold
pub const CALIB_T2: f64 = /* tuned */;    // Low ↔ VeryLow threshold
pub const CALIB_ALPHA: f64 = /* tuned */; // position_criticality near-extreme boost
pub const ANALYSIS_VERSION: u32 = 1;
```

Exact numeric values are outputs of the §4.2 calibration procedure, committed as part of Commit 1 (§9). Each constant's doc-comment cites the SHA of `baseline_heuristic.json` it was derived from.

### 4.2 Procedure

**Six steps, run once pre-commit and re-run any time `ANALYSIS_VERSION` bumps.**

**Step 0 — Pre-inversion input audit.** Enumerate every input field the pre-refactor `cpdt_precision.rs` tier-assignment reads. Method: `grep` the file for field accesses against `AppliedLayer`, `AppliedPlan`, `PrecisionConfig`, and any module-state types the tier-assignment function reaches.

- **Expected finding:** input set = `{layer_index, element_count, layer_kind}`. The layer-kind signal is already factored out as the §3.6 hard override; the remaining inputs `{layer_index, element_count}` are exactly what the formula's non-weight-dependent factors read.
- **If Step 0 surfaces any additional input** (parent module name, bias presence, quantization-aware flags, WGGO state), the finding is a blocker:
  - *Preferred resolution:* factor the new input into the unified scorer's formula as an explicit factor, document it in the spec, re-run calibration.
  - *Fallback resolution:* replace Step 1's closed-form inversion with grid search over the full input space; accept that the Step 5 100% byte-identity target drops to a justified `<100%` threshold documented alongside the input audit.

**The Step 0 artifact is committed** at `tools/cpdt_calibrate_audit.md`. It is SHA-anchored: the audit names the exact commit of `cpdt_precision.rs` against which the `grep` was run, so future readers can reproduce the audit via `git show <sha>:crates/nsl-codegen/src/cpdt_precision.rs`. Template:

```
# CPDT Calibrate Audit

**Audited file:** crates/nsl-codegen/src/cpdt_precision.rs
**File commit SHA:** <sha>
**Audit performed:** <date>
**Grep invocation:** <command>

## Enumerated Inputs
- [ ] layer_index (used at: line X)
- [ ] element_count (used at: line Y)
- [ ] layer_kind (used at: line Z)
- [ ] <any other surfaced field>

## Conclusion
Input set: {<enumerated>}
Closed-form inversion: [sound | unsound; justification]
```

**Step 1 — Snapshot the current heuristic.** Run the pre-refactor `cpdt_precision.rs` tier assignment against `{calib_tiny, calib_small, calib_medium}` and serialize results to `tests/fixtures/cpdt_calibration/baseline_heuristic.json` (committed). This is the byte-identity target for the no-weights path.

**Step 2 — Closed-form inversion.** Given Step 0's confirmation that inputs are `{layer_index, element_count, layer_kind}`, invert the heuristic algebraically: for each tier-boundary decision in the pre-refactor heuristic, solve for `(CALIB_K, CALIB_T0, CALIB_T1, CALIB_T2)` such that the formula-computed score produces the same tier label on every non-kind-overridden parameter in the baseline snapshot.

**Step 3 — Grid search for underdetermined parameters.** The near-extreme boost `CALIB_ALPHA` is not determined by closed-form inversion (it doesn't affect first/last/middle layers, and calibration fixtures with `L ≥ 4` provide additional constraints). Grid search over `α ∈ {0.0, 0.1, 0.2, ..., 1.0}`; pick the value that maximizes byte-identity agreement on `calib_small` (L=8) and `calib_medium` (L=16). Ties broken toward the middle of the grid (prefer `α = 0.5` or adjacent).

**Step 4 — Commit constants** as `pub const`s in `cpdt_sensitivity.rs` with doc-comments citing the `baseline_heuristic.json` commit SHA.

**Step 5 — Verification.** Run unified scorer on `{calib_*, weights=None}`; assert 100% byte-identity against `baseline_heuristic.json`. Running the calibration script `tools/cpdt_calibrate.rs` (Commit 1) produces the constants + verifies the identity in one command.

### 4.3 `CPDT_CALIB_K` Escape Hatch

Environment-variable override for non-standard weight distributions (e.g., muP-trained models, quantization-aware-trained checkpoints with unusual magnitude patterns). Set as `CPDT_CALIB_K=0.05` before invoking `nsl build`.

**Interaction with weights:** `CPDT_CALIB_K` affects only the no-weights path of `gradient_magnitude_est`. When weights are present (via `--weights` or auto-detected AST `load_safetensors(...)`), the computed RMS value is authoritative and `CPDT_CALIB_K` is ignored.

If `CPDT_CALIB_K` is set while weights are present, a single stderr warning is emitted at plan time:

```
warning: CPDT_CALIB_K=<value> is set but ignored.
         Weights are present, so computed gradient_magnitude_est is authoritative.
         If you intended CPDT_CALIB_K to apply, unset --weights or remove the
         AST load_safetensors reference. If the env var is vestigial, you can
         ignore this warning or unset it.
```

Silent-ignoring is a footgun. The expanded warning tells the user what happened, why, and how to act on it without guessing.

### 4.4 `tools/cpdt_calibrate.rs`

Dev-only binary, gated behind `[features] calibrate = []` in `crates/nsl-codegen/Cargo.toml` so it doesn't land in release builds. Runnable via `cargo run --features calibrate --bin cpdt_calibrate`. Single command: regenerates all five constants from `baseline_heuristic.json`, verifies 100% byte-identity, prints a diff-ready constants block for copy-pasting into `cpdt_sensitivity.rs`.

---

## 5. Fixtures & Snapshots

Three synthetic fixtures, deterministic from a fixed seed, cover the tensor-shape regimes Phase 1 cares about: square attention projections, tall FFN projections, wide FFN projections, norms, and embeddings.

### 5.1 Fixture Table

| Fixture | Layers (L) | d_model | d_ffn | Ratio | vocab | Tied emb? | Bias? | Storage |
|---|---|---|---|---|---|---|---|---|
| `calib_tiny` | 2 | 128 | 512 | 4.0× | 256 | No | None | Committed f32, ~2.2 MB |
| `calib_small` | 8 | 512 | 1792 | 3.5× (non-4× deliberately) | 8192 | Yes | None | Committed **f16**, ~68 MB |
| `calib_medium` | 16 | 1024 | 4096 | 4.0× | 32768 | No | Mixed (half-and-half) | Regen-at-test-time, ~600 MB f16 in `target/` |

**Note on `calib_small` storage:** the user's initial approval assumed ~30 MB committed. Computed size at f32 is ~138 MB (8 × 3.8M params/layer + 4.2M tied embedding = 34.6M × 4 B ≈ 138 MB). Storing `calib_small.safetensors` as f16 reduces this to ~68 MB — still larger than the initial estimate but within tolerable committed-repo-size bounds (`.gitattributes` handles it without LFS). **The scorer formula `||W||_F / sqrt(numel)` is dtype-agnostic at f64 computation precision**, so f16 storage does not affect calibration correctness. The fixture generator's doc-comment notes this dtype choice explicitly.

**Why these three:**
- `calib_tiny` (L=2): minimum coverage of all five shape classes, runs in <1s on CPU, exercises the small-`L` degeneracy path in `position_criticality`.
- `calib_small` (L=8, tied embeddings, 3.5× FFN ratio): tests tied-weight parameter sharing and catches ratio-dependent bugs that a 4×-only corpus would miss.
- `calib_medium` (L=16, 32K vocab, mixed bias): tests realistic transformer scale and both bias-present and bias-absent layer variants.

### 5.2 Fixture Generation

**Seed:** `rand::rngs::StdRng::seed_from_u64(0xC9D7DA7ACA15B)` — 13 hex digits = 52 bits, fits u64. Set once per fixture, reset between fixtures so they are independently reproducible.

**Init scheme:** NSL-native defaults as shipped by `nsl.nn.Linear`, `nsl.nn.Embedding`, `nsl.nn.LayerNorm` at Phase 1 commit time. The concrete schemes are read from the stdlib source and pinned in the fixture generator's doc-comment along with the exact source SHA, so future reproduction works even if stdlib init changes. Rationale for native defaults (not HF or PyTorch conventions): fixtures should reflect what NSL users actually get when they declare models.

**Pinned crate versions:** `rand` and `safetensors` pinned in the fixture generator's `Cargo.toml`. Upgrading either requires regenerating committed fixtures and re-running calibration — documented as a manual maintenance step, not automatic.

**Generator location:**

```
crates/nsl-codegen/tests/fixtures/cpdt_calibration/
├── generate.rs                  — binary target, regenerates all three fixtures
├── calib_tiny.safetensors       — committed, f32, ~2.2 MB
├── calib_small.safetensors      — committed, f16, ~68 MB
├── calib_medium_regen.rs        — helper called by test harness; regens calib_medium into target/
├── baseline_heuristic.json      — §4 Step 1 snapshot
└── expected_weights_present.json — §5.3 regression snapshot
```

### 5.3 `expected_weights_present.json` — Generation Workflow

**Role:** regression snapshot, not correctness oracle. Captures the scorer's weights-present tier assignments at ship time; future changes to the scorer that produce different tiers fail the snapshot-equality test, drawing attention to the change. Does **not** prove the captured tiers are semantically correct — that is the adversarial fixture's job (§6).

**Generation workflow** (addresses the red-test ordering concern from Section 3 review):

- **Commit 3** commits `expected_weights_present.json`.
- **The JSON is generated by a local prototype of the Commit 4 implementation** (`gradient_magnitude_est` reading weights) — not committed.
- The prototype implementation MUST match the final Commit 4 implementation byte-for-byte. If they diverge at Commit 4 landing time, Commit 4's test fails; the fix is either updating the JSON (if Commit 4 is correct) or fixing Commit 4 (if the prototype was the reference).
- Commit 3's message names the prototype approach explicitly so future readers don't wonder how the expected output predates the code that produces it.

### 5.4 Disagreement Metric for Regression Test

The 95%-agreement gate between no-weights and weights-present paths on the baseline corpus uses **parameter-count-weighted disagreement**, not layer-count:

```rust
let disagreement_fraction: f64 =
    applied.layers().filter(|l| tier_no_weights[l] != tier_weights[l])
                    .map(|l| l.numel() as f64)
                    .sum::<f64>()
    /
    applied.layers().map(|l| l.numel() as f64).sum::<f64>();

assert!(disagreement_fraction < 0.05);
```

**Rationale:** a tier disagreement on a 100M-element FFN matrix is a much larger plan-output impact than a disagreement on a 64-element bias vector. The unweighted metric (fraction of layers disagreeing) would overweight small parameters that don't affect the optimizer-memory budget meaningfully.

**Zero-numel edge case:** the metric treats zero-numel layers as contributing zero to the disagreement fraction — arguably correct (a vacuous layer's tier doesn't affect the plan) but could hide a scoring bug that systematically mistags vacuous layers. Phase 1 fixtures have no zero-numel layers; if future fixtures do, the metric may need revision.

---

## 6. Adversarial Fixture

**Purpose:** correctness gate that proves the weight-aware path is doing real work, not silently matching the no-weights path because of a calibration-too-generous or silent-stub bug. Equivalent to the gate=1.0 fixture from WRGA B.3.1.

**Catches three bug classes in one fixture:**
1. **Silent-stub:** `gradient_magnitude_est` always returns `CALIB_K` regardless of `WeightEntry`. Target layer tier doesn't move.
2. **Mis-calibration:** target layer moves but lands at the wrong tier. Caught by the exact-equality assertion, not a strictly-higher inequality.
3. **Cross-layer contamination:** scorer broadcasts weight effects globally. Caught by the all-other-layers-unchanged assertion.

### 6.1 Construction (in-test, not committed separately)

```rust
#[test]
fn adversarial_weight_aware_localized_tier_shift() {
    // Precondition 1: target layer is not kind-overridden.
    let target_name = "blocks.4.ffn.down_proj.weight";
    let target_kind = applied.layer_kind(target_name);
    assert!(!is_kind_overridden(target_kind),
        "adversarial target must be a scored (not kind-overridden) layer; \
         current target kind = {target_kind:?}. Revisit fixture selection.");

    // Load the baseline fixture and its expected tier assignments.
    let original = WeightMap::load(CALIB_SMALL_PATH).unwrap();
    let baseline_tiers = load_expected_weights_present_json();

    // Precondition 2: clone-isolation. Mutating the clone must not affect the original.
    let mut clone = original.clone();
    let embed_rms_before = original.get("token_embeddings").unwrap().rms();

    // Compute scaling multiplier from calibrated constants to guarantee
    // the target layer lands exactly at Tier::High.
    let s_pre = scorer.score(&original, target_name);
    let m = (CALIB_T0 / s_pre) * 1.5;  // 1.5× safety margin past T0

    // Apply localized mutation.
    clone.get_mut(target_name).unwrap().scale_in_place(m);

    // Verify clone isolation held.
    assert_eq!(original.get("token_embeddings").unwrap().rms(), embed_rms_before,
        "WeightMap::clone aliased entries; adversarial fixture invariants broken.");

    // Run scorer on mutated clone.
    let adversarial_tiers = scorer.assign_all(&applied, Some(&clone));

    // Strong assertion 1: target lands at Tier::High, exactly.
    assert_eq!(adversarial_tiers[target_name], Tier::High,
        "target layer did not reach Tier::High under {m:.2}× scaling. \
         Likely causes: silent-stub gradient_magnitude_est, mis-calibrated T0, \
         or broken scorer.assign_all path.");

    // Strong assertion 2: every other layer's tier is unchanged.
    for (name, &baseline_tier) in &baseline_tiers {
        if name == target_name { continue; }
        assert_eq!(adversarial_tiers[name], baseline_tier,
            "layer {name} tier changed unexpectedly under localized mutation \
             of {target_name}. Scorer may be broadcasting weight effects across \
             the model.");
    }
}
```

### 6.2 Sample Multiplier Arithmetic

For illustration (exact values depend on calibration output):

- If `CALIB_T0 = 0.50` and target layer's baseline `S_pre ≈ 0.08` (somewhere in Tier 2, between `T2 ≈ 0.04` and `T1 ≈ 0.15`),
- Then `M = (0.50 / 0.08) × 1.5 = 6.25 × 1.5 ≈ 9.4`.

Similar to "10× scaling" in magnitude, but derived from constants rather than arbitrary. The 1.5× safety margin ensures floating-point drift in the scorer doesn't accidentally leave the layer in Tier 1.

### 6.3 Target Layer Choice

`blocks.4.ffn.down_proj.weight` in `calib_small` — middle layer (`l=4` of `L=8`), scored (not kind-overridden), wide-FFN shape class. If a future refactor changes which layers are kind-overridden (e.g., treating `down_proj` as a kind-specific layer), the Precondition 1 assertion fires with a clear "target layer is now kind-overridden; adversarial fixture target selection needs revisiting" message.

---

## 7. Diagnostics, Warnings, Data-Flow Notes

### 7.1 Passive Tier-Agreement Diagnostic

On every CPDT build where weights are present, emit to stderr:

```
[cpdt] weight-aware tier agreement: <X>.<YY>% (<D>/<T> layers, parameter-weighted <W>%)
```

Where `X.YY%` is the raw layer-count agreement, `D/T` is count of disagreeing layers over total, and `W%` is the parameter-weighted agreement fraction (the one that gates the 95% threshold).

**When `W < 95%`:**

```
warning: weight-aware tier agreement below 95% on this model (parameter-weighted
         <W>%). This may indicate that the calibration constants do not fit this
         weight distribution well. Consider filing an issue at <url> referencing
         the post-scoring-normalization deferred work; include the model's weight
         file hash (<sha>) and the full diagnostic output.
```

This is the primary surfacing mechanism for the §1.2 item 5 post-scoring-normalization trigger. Phase 1 does not add `@cpdt(score_normalization=...)`; the diagnostic makes real-model mis-calibration visible so the Phase 2-or-later decision is data-driven rather than user-reported-issue-driven.

### 7.2 Override Warnings

**Case 2 (AST ref + `--weights` flag both present):**

```
warning: --weights <flag_path> override in effect; AST-referenced <ast_path> ignored.
```

**`CPDT_CALIB_K` + weights:** see §4.3 (expanded message with resolution hint).

### 7.3 `cpdt_joint.rs` NOTE Comment

The Commit 1 refactor adds an inline comment at the top of `cpdt_joint::solve`:

```rust
pub fn solve(input: JointInput) -> JointPlan {
    // NOTE: If this function is modified to read per-parameter fields from
    // input.precision (scores, tiers, layer-specific data), the CPDT calibration
    // contract tightens — see cpdt_sensitivity.rs ANALYSIS_VERSION rule. Today,
    // this function reads only aggregate fields (total_optim_bytes, etc.); the
    // tier-assignment byte-identity regression gate is sufficient because
    // aggregates are derived from tier labels. Reading per-parameter fields
    // directly would require a stricter calibration contract that preserves
    // per-parameter score magnitudes, not just tier labels.
    // ...
}
```

Finding (verified Commit 1 prep): as of **main @ `52ab261`** (the pre-refactor baseline), `cpdt_joint::solve` reads exactly `input.precision.total_optim_bytes` and no per-parameter fields. Consequence: byte-identity of tier assignments on the baseline corpus implies byte-identity of `total_optim_bytes` implies byte-identity of joint-solver output. **No additional regression gate needed beyond the tier-assignment byte-identity gate.**

**Joint solver determinism check (Commit 1 prep):** verify that `cpdt_joint::solve` has no non-determinism sources (random seeds, hash-map iteration order in internal data structures, time-dependent early termination). If Commit 1 prep finds any, an explicit joint-solver snapshot test is added to Commit 1's test set rather than relying on the transitive argument.

### 7.4 CPDT Invariants File

Phase 1 establishes invariants that must survive Phase 2 and beyond. Committed in Commit 6 at `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_cpdt_weight_aware_invariants.md` (new, mirroring `project_wrga_fused_ptx_rewrite.md`'s structure). Numbering starts at #1.

Initial Phase 1 invariants (exact list + justifications in the file itself):

1. `ANALYSIS_VERSION` bumps on every formula/factor/tier-boundary change.
2. `cpdt_joint.rs::solve` reads only tier-derived aggregates, never per-parameter fields.
3. Layer-kind override (Norm, Embedding) takes precedence over formula score.
4. `SensitivityScorer` is a concrete type, not a trait.
5. Rank-normalization is deferred; factors output physical units.
6. `--weights` flag overrides AST format inference entirely; sources are never combined.
7. No-weights path matches baseline heuristic byte-identical on calibration corpus (100%).
8. Weights-present path agrees with no-weights path within 5% parameter-weighted on baseline corpus.
9. Scorer state derives exclusively from `WeightMap` read-only access; no shared computations with sparsity / constant-folder / dead-weight passes.

---

## 8. Commit Sequence

Six commits mirror the WRGA B.3.1 pattern (refactor → primitives → red → green → integration → close-out), each with its own independently-verifiable acceptance check.

### 8.1 Commit 1 — `refactor(cpdt): extract unified SensitivityScorer`

**Scope:**
- Delete `crates/nsl-codegen/src/cpdt_precision.rs`. Tier-assignment code migrates into new `cpdt_sensitivity.rs`; tier-application utilities migrate into new `cpdt_tier_apply.rs` via `git mv` + import updates.
- Commit `cpdt_sensitivity.rs` with formula, `ANALYSIS_VERSION` constant, all five calibration constants, layer-kind override, and module doc-comment stating the bump rule.
- Commit `tools/cpdt_calibrate.rs` (dev-only binary, gated behind `[features] calibrate = []`) and `tools/cpdt_calibrate_audit.md` (SHA-anchored audit artifact).
- Commit `tests/fixtures/cpdt_calibration/baseline_heuristic.json`.
- Add inline `// NOTE:` comment to `cpdt_joint::solve` per §7.3.
- Run Commit 1 prep check: confirm `cpdt_joint::solve` determinism. If non-deterministic, add an explicit joint-solver snapshot test to this commit's test set.
- `gradient_magnitude_est` returns `CALIB_K` unconditionally. No weight-reading code yet.

**Acceptance:**
- `cargo build -p nsl-codegen` green; `cargo test -p nsl-codegen` green.
- On `{calib_tiny, calib_small, calib_medium}` the scorer output matches `baseline_heuristic.json` byte-identical (100%).
- `cargo run --features calibrate --bin cpdt_calibrate` reproduces the committed constants.
- `cpdt_joint.rs` behavior byte-identical to pre-refactor (explicit snapshot if needed per prep check).

### 8.2 Commit 2 — `test(cpdt): sensitivity-scorer primitive unit tests`

**Scope:** Structural tests for each primitive:
- `position_criticality` at `L ∈ {1, 2, 3, 4, 8, 16}` covering first, last, near-extreme, middle, and `L < 4` degeneracy branches.
- `element_count` extraction from `AppliedLayer`.
- `gradient_magnitude_est` with `Option<&WeightEntry> = None` → returns `CALIB_K` exactly.
- `assign_tier` boundary behavior at `T0 ± ε`, `T1 ± ε`, `T2 ± ε`.
- Layer-kind override: `{Norm, Embedding}` → `Tier::High` regardless of score.

**Acceptance:** all new unit tests green; no existing tests regress.

### 8.3 Commit 3 — `test(cpdt): adversarial + weights-present fixtures (red)`

**Scope:**
- Add the §6 adversarial fixture test (with clone-isolation and kind-override preconditions).
- Add the §5 weights-present regression test.
- Commit `expected_weights_present.json` generated by **a local prototype of Commit 4's implementation** (not committed). Commit message states: "Expected JSON was generated from a local prototype matching Commit 4's implementation byte-for-byte. If Commit 4 diverges from the prototype, Commit 4's test will fail and the expected JSON will need an update."
- Mark both new tests `#[ignore = "red: unblocked by Commit 4"]`.

**Acceptance:** tests compile green, run ignored, CI stays green. Tests visible in the PR for review but not executing.

### 8.4 Commit 4 — `feat(cpdt): gradient_magnitude_est reads weights (green)`

**Scope:**
- Implement `gradient_magnitude_est(Some(&WeightEntry)) -> f64` per §3.2 (RMS at f64 precision, single pass).
- Remove `#[ignore]` from Commit 3's two tests.
- Adversarial fixture now passes (target = `Tier::High` exactly, every other layer's tier equals `expected_weights_present.json`).
- Weights-present regression test now passes (snapshot byte-identity against `expected_weights_present.json`).
- Weighted disagreement between no-weights and weights-present paths on baseline corpus: `< 0.05`.

**Acceptance:**
- Commit 3's `#[ignore]` tests flip to green.
- Commit 1's 100% byte-identity on no-weights path is preserved.
- Commit 2's unit tests unchanged.

### 8.5 Commit 5 — `feat(cpdt): --weights / AST auto-detect integration + validation + diagnostics`

**Largest commit in the sequence.** Scope bundles CLI integration, validation pass, diagnostic emission, and warning surfaces. Commit message structures the acceptance check as five explicitly-listed sub-conditions, each independently verifiable:

1. **CLI decision table (§2.1):** AST walk for `load_safetensors(...)`, `--weights` override, absent-both error message with three-option resolution, format-override principle implemented.
2. **`WeightMap` validation pass (§2.2):** shape / dtype / tensor-name cross-check against `AppliedPlan` inside `cpdt::run`, fast error with clear surface.
3. **`CPDT_CALIB_K` + weights warning (§4.3):** noisy-ignoring with expanded resolution-hint text.
4. **Tier-agreement passive diagnostic (§7.1):** emitted on every weights-present CPDT build; `W < 95%` warning points at post-scoring-normalization deferred work.
5. **File-not-found handling (§2.3):** `--weights <nonexistent>` produces clear error with the attempted path shown, not the vague `WeightMap::load` internal error.

**Acceptance (five independently-verifiable tests):** one per sub-condition above, each listed as a separate test row in §9.

### 8.6 Commit 6 — `docs(cpdt): Phase 1 close-out + Phase 2 stub + invariants`

**Scope:**
- Update `docs/superpowers/specs/` to mark Phase 1 shipped.
- Write Phase 2 stub doc at `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md` per §12.
- Write `project_cpdt_weight_aware_invariants.md` in memory per §7.4.
- Update `MEMORY.md` with Phase 1 completion entry + pointer to Phase 2 trigger conditions. Keep under 200 lines; extract topic files if needed.
- Retire `cpdt_precision.rs` references in existing docs.

**Acceptance:** all linked docs reachable; no broken references. `MEMORY.md` ≤ 200 lines or topic-file extraction applied.

---

## 9. Test Matrix

12 test surfaces across 6 commits. Each commit adds or flips exactly the tests listed for it.

| # | Test | Type | Fixture | Commit | Catches |
|---|------|------|---------|--------|---------|
| 1 | Primitive unit tests (`position_criticality` at `L ∈ {1..16}`, `element_count`, `gradient_magnitude_est` null, `assign_tier` boundaries, layer-kind override) | Unit | inline | C2 | Formula-level regressions |
| 2 | `baseline_heuristic.json` byte-identity on no-weights path | Snapshot | `calib_{tiny, small, medium}` | C1 (committed); asserted C1 onward | Refactor drift |
| 3 | `expected_weights_present.json` byte-identity on weights-present path | Snapshot | `calib_{tiny, small, medium}` | C3 (committed); asserted C4 onward | Scorer drift post-landing |
| 4 | Weighted disagreement metric (`< 0.05`) | Integration | `calib_{tiny, small, medium}` | C4 | Calibration drift between no-weights/weights paths |
| 5 | Adversarial fixture: scale target × M, assert `Tier::High` + every other layer unchanged + clone-isolation precondition + kind-override precondition | Correctness gate | `calib_small` (in-memory-mutated clone) | C3 (red), C4 (green) | Silent-stub, mis-calibration, cross-layer contamination, clone aliasing, future kind-override regression |
| 6 | AST `load_safetensors(...)` auto-detect (Case 1) | Integration | tiny `.nsl` fixture | C5 | Four-case table Cases 1 and 2 |
| 7 | `--weights` flag override + stderr warning (Case 2) | Integration | CLI | C5 | Four-case table Cases 2 and 3 |
| 8 | Absent-both error with three-option resolution message (Case 4) | Integration | CLI | C5 | Four-case table Case 4 |
| 9 | `CPDT_CALIB_K` + weights noisy-ignoring warning | Integration | env var + CLI | C5 | Vestigial env-var surfacing |
| 10 | Tier-agreement diagnostic output (line emitted; `W < 95%` → warning) | Integration | representative weights-present build | C5 | Passive 5%-trigger monitoring |
| 11 | `WeightMap` validation: shape / dtype / tensor-name mismatches produce clear fast errors | Integration | weight file + declaration | C5 | Weights-vs-declaration drift |
| 12 | `--weights <nonexistent>` → clear error with path shown | Integration | CLI | C5 | User-facing ergonomics (typo'd path, missing file, permissions) |

---

## 10. Close-Out Criteria

Phase 1 is shipped when all of the following are true:

1. **Structural:**
   - `cpdt_precision.rs` deleted.
   - `cpdt_sensitivity.rs` owns scoring + tier assignment.
   - `cpdt_tier_apply.rs` owns tier → IR emission utilities.
   - `ANALYSIS_VERSION = 1` constant landed with the commit-rule doc-comment.
2. **Correctness:**
   - Commit 4's adversarial fixture passes with exact `Tier::High` assertion + all-other-layers-unchanged + both preconditions (clone-isolation, not-kind-overridden).
   - `baseline_heuristic.json` 100% byte-identity on no-weights path.
   - `expected_weights_present.json` byte-identity on weights-present path.
   - Weighted disagreement `< 0.05` on baseline corpus.
3. **Integration:**
   - All four decision-table cases covered by CLI tests (rows 6, 7, 8, 12).
   - `WeightMap`-vs-`AppliedPlan` validation pass inline (row 11).
   - `CPDT_CALIB_K` + weights noisy-ignoring warning emitted (row 9).
   - Tier-agreement diagnostic on every weights-present build (row 10).
4. **Docs:**
   - Phase 2 stub doc committed.
   - `MEMORY.md` has Phase 1 completion entry.
   - CPDT invariants file committed with ≥9 initial Phase 1 entries (§7.4).
   - No orphan references to `cpdt_precision.rs`.
5. **Calibration reproducibility:**
   - `tools/cpdt_calibrate.rs` binary runnable; reproduces committed constants.
   - `tools/cpdt_calibrate_audit.md` committed with SHA-anchored grep output.
6. **No regressions:**
   - Existing CPDT pipeline integration tests (PR #37) pass unchanged.
   - `cpdt_joint.rs` output byte-identical on baseline corpus (transitive via tier-assignment byte-identity + `total_optim_bytes`-only reading; explicit snapshot test if Commit 1 prep finds determinism concerns).

---

## 11. Phase 2 Stub

Short doc committed in Commit 6 at `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md`. The stub is a pointer, not a design — Phase 2's full design runs through its own brainstorming cycle when triggered.

### 11.1 Measurement Trigger

Phase 2 work is scheduled when **either**:

- **(a)** Weighted disagreement on a shipped NSL model's weight file exceeds 5% against an extended baseline corpus — measurable automatically via the §7.1 `[cpdt] weight-aware tier agreement: X%` diagnostic emitted on every CPDT build. When `W < 95%` fires repeatedly on real-model builds, the trigger is hit.
- **(b)** A user-filed issue demonstrates a workflow requiring cross-checkpoint tier distribution comparison. This unblocks the `@cpdt(score_normalization=...)` deferred item from §1.2.

### 11.2 Scope

- **Spectral condition computation.** Randomized SVD or power iteration on `W^T W` — implementation method deferred to Phase 2 design. Tradeoff: compile-time cost vs numerical accuracy.
- **Sidecar cache reading/writing** — mirror `wggo_weight_analysis_cache.rs` exactly:
  - Filename suffix: `.cpdt-sensitivity.json`
  - Key format: `(sha256, tensor_name, analysis_version)` where `analysis_version == ANALYSIS_VERSION`
  - Format: JSON with explicit schema version field
  - Cache-miss / version-mismatch / corruption: silent fallback to fresh computation (best-effort)
- **CI check that enforces `ANALYSIS_VERSION` bump** on any diff touching `fn compute_score`, `fn assign_tier`, `fn gradient_magnitude_est`, or `fn position_criticality` — grep-based, runs in CI.

### 11.3 Inherited Discipline (Non-Negotiable Phase 1 Commitments)

- Six-commit structure: refactor → primitives → red → green → integration → close-out.
- Byte-identity regression gate on existing no-weights and weights-present paths.
- Adversarial fixture mandatory for any new factor added to the scorer (each new factor gets its own in-test mutation proving it's doing real work).
- Scorer remains a concrete type, not a trait. Spectral is a new factor *field* in `SensitivityScorer`, not a new `Scorer` implementation.
- Rank-normalization remains out of scope; any need for cross-checkpoint normalization is a separate `@cpdt(score_normalization=...)` opt-in.

### 11.4 Pre-Decided Architectural *Direction* (Verify Before Committing)

Phase 2 **aims** to add spectral as a fourth multiplicative factor without retuning `T0, T1, T2`. This requires the measured spectral distribution on typical matrices to have geometric mean ≈ 1.0.

**Phase 2's first commit verifies this assumption** on the calibration corpus. If the assumption fails (geometric mean diverges from 1.0 by more than, say, 2× across `calib_{tiny, small, medium}`), Phase 2's scope expands to include **either** (i) a multiplicative spectral normalization that enforces the ≈1.0 property by construction, **or** (ii) retuning `T0/T1/T2` against the extended four-factor baseline. Phase 1 does not pre-commit to either resolution — this is a precondition-check, not a pre-decided choice.

### 11.5 Open Questions Carried into Phase 2 Design

1. **Cache-miss behavior.** Does a miss trigger inline spectral computation (slow, consistent with cache-present plan) or fall back to three-factor scoring (fast, different plan)? Phase 1 does not answer this; Phase 2 must.
2. **SVD implementation tradeoffs.** Randomized SVD vs power iteration; compile-time cost vs numerical accuracy; whether an approximate-spectral-below-threshold shortcut is acceptable.
3. **Sidecar cache reusability.** Should the cache file be reusable by other weight-consuming passes (ZK analysis, WGGO importance), or is per-pass sidecar isolation better? The WGGO cache is pass-specific (`.wggo-importance.json`); CPDT's cache being pass-specific (`.cpdt-sensitivity.json`) inherits that pattern. Consolidation to a single multi-scope cache file is a separate refactor.

---

## 12. Open Questions Carried Into Implementation

None that block landing. Two items to resolve during Commit 1 prep:

- **Joint solver determinism verification.** Confirm no non-determinism sources (random seeds, hash-map iteration order, time-dependent early termination) in `cpdt_joint::solve`. If any exist, add an explicit snapshot test to Commit 1.
- **NSL stdlib init schemes snapshot.** Read `nsl.nn.Linear`, `nsl.nn.Embedding`, `nsl.nn.LayerNorm` init implementations at Phase 1 commit time; pin exact SHA and scheme description in the fixture generator's doc-comment.

---

## 13. Summary

Phase 1 ships a unified `SensitivityScorer` that assigns precision tiers to every parameter from three cheap factors. When weights are present, tiers reflect real tensor magnitudes; when absent, tiers match the pre-refactor heuristic byte-identical. The six-commit sequence (refactor → primitives → red → green → integration → close-out) lands through a tested, reviewable path. The adversarial fixture is the correctness gate; `expected_weights_present.json` is the regression gate; the passive tier-agreement diagnostic is the forward-looking signal that tells us when Phase 2 becomes necessary.

The design trades Phase 2 cost (spectral computation + caching) for Phase 1 simplicity (no expensive SVD, no cache infrastructure). The cache architecture is committed at the spec level now — `ANALYSIS_VERSION` constant, the bump rule, the `.cpdt-sensitivity.json` suffix, the SHA-keying — so Phase 2 adds the implementation without reinventing the schema. That's the iii unified-scorer pattern extended to architecture: one decision-set, verified in Phase 1, inherited cleanly by Phase 2.
