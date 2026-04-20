# CPDT Weight-Aware Phase 2 — Stub (not a design)

> **Status:** scheduled, not active. This is a pointer for when the measurement
> trigger fires or a user workflow requires it. Phase 2's full design runs
> through its own brainstorming cycle when that happens.

**Parent spec:** [2026-04-18-cpdt-weight-aware-phase1-design.md](2026-04-18-cpdt-weight-aware-phase1-design.md) §11 (Phase 2 Stub) — the primary source; this file repeats it so memory/index links don't traverse two hops.

**Phase 1 invariants that must survive:** see `project_cpdt_weight_aware_invariants.md` in the session memory directory.

---

## Measurement Trigger

Phase 2 is scheduled when **either**:

- **(a)** Weighted disagreement on a shipped NSL model's weight file exceeds 5% against an extended baseline corpus — measurable automatically via the `[cpdt] weight-aware tier agreement: X%` diagnostic emitted from `invoke_cpdt_if_enabled` on every CPDT build. When `param_pct < 95%` fires repeatedly on real-model builds, the trigger is hit.
- **(b)** A user-filed issue demonstrates a workflow requiring cross-checkpoint tier distribution comparison. This unblocks the `@cpdt(score_normalization=...)` deferred item from the Phase 1 spec §1.2.

## Scope

- **Spectral condition computation** as a fourth multiplicative factor in `SensitivityScorer`. Randomized SVD or power iteration on `W^T W` — implementation method deferred to Phase 2 design. Compile-time cost vs numerical accuracy is the live tradeoff.
- **Sidecar cache reader/writer**, modeled on `wggo_weight_analysis_cache.rs`:
  - Filename suffix: `.cpdt-sensitivity.json`
  - Key format: `(sha256, tensor_name, analysis_version)` where `analysis_version == ANALYSIS_VERSION`
  - Format: JSON with explicit schema-version field
  - Cache miss / version mismatch / corruption: silent fallback to fresh computation (best-effort)
- **CI check that enforces `ANALYSIS_VERSION` bumps** on any diff touching `fn compute_score`, `fn assign_tier`, `fn gradient_magnitude_est`, or `fn position_criticality`. Grep-based, runs in CI.

## Inherited Discipline (Non-Negotiable Phase 1 Commitments)

- Six-commit structure: refactor → primitives → red → green → integration → close-out.
- Byte-identity regression gate on existing no-weights and weights-present paths.
- Adversarial fixture mandatory for any new factor added to the scorer (each new factor gets its own in-test mutation proving it's doing real work).
- Scorer remains a concrete type, not a trait. Spectral is a new factor *field* in `SensitivityScorer`, not a new `Scorer` implementation.
- Rank-normalization remains out of scope; any need for cross-checkpoint normalization is a separate `@cpdt(score_normalization=...)` opt-in.

## Pre-Decided Architectural Direction (Verify Before Committing)

Phase 2 **aims** to add spectral as a fourth multiplicative factor without retuning `T0, T1, T2`. This requires the measured spectral distribution on typical matrices to have geometric mean ≈ 1.0.

**Phase 2's first commit verifies this assumption** on the calibration corpus. If the assumption fails (geometric mean diverges from 1.0 by more than ~2× across `calib_{tiny, small, medium}`), Phase 2's scope expands to include **either**:

1. a multiplicative spectral normalization that enforces the ≈1.0 property by construction, or
2. retuning `T0/T1/T2` against the extended four-factor baseline.

Phase 1 does not pre-commit to either resolution — this is a precondition check, not a pre-decided choice.

## Open Questions Carried Into Phase 2 Design

1. **Cache-miss behavior.** Does a miss trigger inline spectral computation (slow, consistent with cache-present plan) or fall back to three-factor scoring (fast, different plan)? Phase 1 does not answer this; Phase 2 must.
2. **SVD implementation tradeoffs.** Randomized SVD vs power iteration; compile-time cost vs numerical accuracy; whether an approximate-spectral-below-threshold shortcut is acceptable.
3. **Sidecar cache reusability.** Should the cache file be reusable by other weight-consuming passes (ZK analysis, WGGO importance), or is per-pass sidecar isolation better? The WGGO cache is pass-specific (`.wggo-importance.json`); CPDT's cache being pass-specific (`.cpdt-sensitivity.json`) inherits that pattern. Consolidation to a single multi-scope cache file is a separate refactor.

## Phase 1 Deferred Items Flagged For Phase 2 (or earlier cleanup)

Items Phase 1 Commit 5 scope-reduced; record kept here so they surface during Phase 2 planning:

- **AST `load_safetensors(...)` auto-detect** in nsl-cli — pure ergonomics; not gated by Phase 2 measurement trigger; can land any time as a standalone pass.
- **`@cpdt(weight_aware=false)` runtime opt-out** — semantic field exists, codegen threading deferred. Required if any Phase 2 consumer needs an escape hatch at build time.
- **`cpdt_sensitivity::validate(wm, applied)` body** — partially landed 2026-04-20 as layer-prefix validation. Design: `docs/superpowers/specs/2026-04-20-cpdt-validate-body-design.md`. Catches "wrong checkpoint entirely" (HuggingFace-format for NSL-native model) via per-layer prefix match on `blocks.N` / `layers.N` / `h.N`. **Per-tensor shape/dtype validation remains deferred to Phase 2** where the spectral-factor wiring produces the per-tensor metadata pipeline anyway. When Phase 2 lands, validate extends to cross-check `AppliedLayer`-declared tensor names against `WeightMap` entries, activating the currently-unused `MissingTensor` / `ShapeMismatch` / `DtypeMismatch` variants. Phase 1 layer-prefix runs first as the fast-path check; per-tensor validation runs only if layer-prefix passes.
- **Full `nsl-cli/tests/cpdt_weights_cli.rs` five-case decision-table test suite** — the tier-agreement diagnostic is already covered by `crates/nsl-codegen/tests/cpdt_tier_agreement.rs`; CLI tests are belt-and-suspenders. Land with the AST auto-detect.
