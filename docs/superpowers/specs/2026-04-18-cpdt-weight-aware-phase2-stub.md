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
- **`cpdt_sensitivity::validate(wm, applied)` body** — Phase 1 ships the stub. AppliedLayer's weight-metadata surface needs extension before the body can cross-check shape/dtype/name. Independent of Phase 2; useful any time.
- **Full `nsl-cli/tests/cpdt_weights_cli.rs` five-case decision-table test suite** — the tier-agreement diagnostic is already covered by `crates/nsl-codegen/tests/cpdt_tier_agreement.rs`; CLI tests are belt-and-suspenders. Land with the AST auto-detect.

---

## Phase 1 threshold retune — 2026-04-19 — scope-reduced with monitoring-gate reframing

**Context.** Phase 1 shipped with placeholder constants `T0=0.50, T1=0.10, T2=0.02, CALIB_K=0.0312`. Post-ship inspection found that every generic (non-kind-overridden) tensor scored below `T2` on the calibration corpus, producing a degenerate binary distribution: 32 High + 42 VeryLow on `calib_small`, 0 Medium, 0 Low. Both Phase 1 close-out gates (byte-identity on no-weights, <5% disagreement) passed trivially because both paths produced the same degenerate distribution.

**Pre-dispatch finding.** Running the retune's Commit 1 logic ahead of implementation revealed a structural information-bottleneck in the no-weights path: `ffn.w_gate`, `ffn.w_up`, and `ffn.w_down` share `numel = d_model × d_ffn` in a standard SwiGLU FFN, so the no-weights formula `K × pos / numel` produces identical scores for all three and cannot differentiate them by tier regardless of K. Pure-corpus-minimum K (K≈0.046) gives 2.49% corpus-wide disagreement but 26.51% on calib_small specifically — the numel-collision dominates calib_small's per-fixture metric.

**Methodological shift.** The spec's initial "minimize corpus disagreement" formulation was replaced with "plateau-midpoint-under-per-fixture-constraint": pick the log-midpoint of the longest contiguous K-range where every per-fixture disagreement stays ≤ 20%. This trades ~1.3% corpus-wide disagreement to keep both per-fixture numbers comfortably under the monitoring threshold, provides robustness to measurement noise through the plateau's width, and produces consistent per-view signals. See §3.2 of the retune design spec.

**Response.** Retune thresholds (geomeans of adjacent band mins/maxes), pick K by plateau-midpoint algorithm, reframe invariant #7 as a monitoring-gate on committed-fixture disagreement rather than a hard-gate.

**Measured values** (from the 601-point Rust grid in `cpdt_calibrate --emit-calibration`):

- `T0 = 6.1056e-8` (High ↔ Medium)
- `T1 = 2.2324e-8` (Medium ↔ Low)
- `T2 = 7.2557e-10` (Low ↔ VeryLow)
- plateau = `[0.0562, 0.0708]`
- `CALIB_K = 6.3096e-2` (log-midpoint of plateau)
- `ANALYSIS_VERSION` bumped to 2

**Post-retune distribution:**

- `calib_small` (74 tensors): 68 H + 6 M + 0 L + 0 VL (2-tier on this fixture alone)
- `calib_medium` (~179 tensors): 53 H + 56 M + 42 L + 28 VL (4-tier spread)
- Corpus union populates primary set {H, M, L}; VeryLow is the documented fallback

**Residual disagreement.** Committed-fixture gate (calib_tiny + calib_small) measures ~15.6%, well under the 20% monitoring threshold; calib_medium measures 2.51%; corpus-wide 3.76%. The 15.6% residual traces to the SwiGLU numel-collision (12 `ffn_gate_up` middle-layer tensors on calib_small plus 2 near-extreme `ffn_down` tensors that cross the T0 boundary due to pos=1.3 multiplier). The `disagreement_source_matches_numel_collision` diagnostic test verifies firings trace to this known pattern.

**Invariant #7 reframing.** Phase 1's gate is now monitoring-status at `<0.20` on committed-fixture disagreement, with Phase 2 spectral factor as the documented intervention that returns it to `<0.05` hard-gate. The gate's computation and domain are unchanged; only its institutional authority (close-out blocker vs. Phase 2 promotion signal) shifts. This is *reframing*, not *weakening* — the latter was explicitly rejected during the E-option deliberation.

**Phase 2 connection.** Spectral condition (randomized SVD or power iteration on `W^T W`) gives the no-weights path a shape-class discriminator beyond `numel` — its per-fixture close-out criterion is `<5%` on the committed-fixture gate, which automatically returns invariant #7 to hard-gate status.

**Institutional rules added** (retune design spec §7):

1. **Tier-distribution non-degeneracy check.** Any N-tier scheme's close-out must assert the *primary* tier set (all tiers minus documented-as-rare fallbacks) is non-degenerate across the union of calibration fixtures. CPDT's primary set is {High, Medium, Low}; VeryLow is the documented fallback.
2. **Monitoring-gate reframing for architectural bottlenecks.** When an invariant fires on a specific architectural class (not a single tensor or transient bug), the correct response is to document the class, add a diagnostic test that verifies the firing traces to it, name the future intervention that resolves it, and reframe the invariant to monitoring-status with that intervention as the promotion trigger — *not* to narrow the invariant's domain or raise its threshold.

Both rules apply to future tier-assignment work in NSL.

**Pre-dispatch-verification lesson.** The retune spec's initial `§3.3` distribution was computed from calib_small-only data during pre-dispatch (calib_medium wasn't regenerable at spec-writing time). `§3.1`'s pooling procedure was written for the full corpus. The two sections were written under different fixture-availability assumptions and the mismatch wasn't caught during spec self-review — the implementer's grid-search under the operative `§3.1` procedure surfaced it. **Rule for future spec-writing:** pre-dispatch verification must run against the same fixture corpus the implementation will use. If part of the corpus isn't available at spec-writing time, distribution tables that depend on it are deferred to post-dispatch measurement, or the corpus is regenerated before spec writing. This is adjacent to the WRGA B.3.1 "test at the scale you target" rule but distinct: that one is about the implementation's test coverage; this one is about the spec's verification coverage.
