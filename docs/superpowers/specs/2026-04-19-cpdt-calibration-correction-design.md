# CPDT Phase 1 Threshold Retune — Calibration Correction Design

> **Framing:** This is a post-ship correction to Phase 1, not an evolution. The initial `T0/T1/T2/CALIB_K` constants shipped as placeholders (noted in `cpdt_sensitivity.rs:35-40`); empirical observation after merge shows they produce a degenerate binary tier distribution on the calibration corpus. This spec fixes that without reopening the formula.

**Parent spec:** [2026-04-18-cpdt-weight-aware-phase1-design.md](2026-04-18-cpdt-weight-aware-phase1-design.md)
**Invariants preserved:** `project_cpdt_weight_aware_invariants.md` #1-#9 (all)
**Scope:** two commits on `feat/cpdt-calibration-tune`, no formula change, no CLI change.

---

## 1. Motivation

Phase 1 shipped with `T0=0.50, T1=0.10, T2=0.02, CALIB_K=0.0312`. On the committed calibration corpus:

| Fixture | Kind-overridden → High | Generic → VeryLow | Medium | Low |
|---|---|---|---|---|
| calib_tiny  | 20 | 1  | 0 | 0 |
| calib_small | 32 | 42 | 0 | 0 |

The distribution is degenerate: every generic tensor scores below `T2 = 0.02` and lands `VeryLow`, so Phase 1's 4-tier scheme operates as a 2-tier scheme in practice. Observed generic-score bands on `calib_small`:

- `attn.w{q,k,v,o}`         ~2.4-3.1×10⁻⁷
- `ffn.w_gate`, `ffn.w_up`  ~6.8-8.9×10⁻⁸
- `ffn.w_down`              ~3.6-4.7×10⁻⁸

The degeneracy slipped past Phase 1 close-out because the two shipped gates (byte-identity on no-weights, <5% disagreement) both pass trivially when both paths produce the same binary distribution. Phase 1's close-out did not include a tier-distribution non-degeneracy check; §6 of this spec adds one as institutional discipline for future tier-assignment work.

## 2. Non-Goals

- **No formula change.** `sensitivity = gradient_magnitude × position_criticality / element_count` stays unchanged. The `/numel` divisor encodes "per byte saved by quantizing," which is the correct quantity for a memory-budget-aware tier assignment; softening it replaces this with "per element," a less useful metric for the decision CPDT makes. Phase 2's spectral factor is the right place for finer within-class differentiation.
- **No CLI or `invoke_cpdt_if_enabled` changes.** The threading is correct; only constants move.
- **No weakening of invariant #7.** The <5% disagreement gate stays at 5%; `CALIB_K` is recalibrated so the no-weights path continues to agree with the weights-present path under the new thresholds. Weakening the gate sets the precedent that load-bearing invariants get loosened when inconvenient.

## 3. Design

### 3.1 Threshold computation

Compute `T0, T1, T2` from the observed shape-class score bands on the calibration corpus. Specifically:

1. Load `calib_tiny`, `calib_small`, and regenerate `calib_medium` into `target/cpdt_calibration/` (per the existing regen-at-test-time contract).
2. Score every tensor with the current formula.
3. Classify each **generic** (non-kind-overridden) tensor by name pattern:
   - `attn.w[qkvo]` → band `attn_qkvo`
   - `ffn.w_gate` | `ffn.w_up` → band `ffn_gate_up`
   - `ffn.w_down` → band `ffn_down`
   - Anything else → band `other` (not used for threshold placement)
4. For each band, record `(min, max, geometric_mean)` across all three fixtures.
5. **Cross-fixture consistency guard.** Verify the three bands are ordered consistently across fixtures (`attn_qkvo > ffn_gate_up > ffn_down`). If the ordering differs on any fixture, exit the calibration binary non-zero with a diagnostic naming the offending fixture and bands. This is the equivalent of the "bands inconsistent" guard for fixture-generalization.
6. Compute:
   - `T0 = sqrt(attn_qkvo.min × ffn_gate_up.max)` — boundary such that attn → High
   - `T1 = sqrt(ffn_gate_up.min × ffn_down.max)` — boundary such that ffn_gate/up → Medium
   - `T2 = sqrt(ffn_down.min × CALIB_T2_FLOOR)` where `CALIB_T2_FLOOR = 1e-10`
7. Emit the diff-ready Rust constants block to stdout.

Expected output (concrete values come from the binary; spec lists approximate targets for review):
- `T0 ≈ 1.3×10⁻⁷`, `T1 ≈ 5.9×10⁻⁸`, `T2 ≈ 6×10⁻¹⁰`

### 3.2 CALIB_K recalibration

For each band, compute the `CALIB_K` value that would place the no-weights score (`CALIB_K × pos / numel`) in the same tier the weights-present path produces for that band's median tensor:

- Per-band candidate: `K_band = target_score_band × numel_median / pos_median`
- The target score for each band is the geometric mean of that band's `[min, max]` range on the weights-present path.
- `pos_median` = 1.0 for middle layers in `calib_small` (L=8, layers 2-5 are middle), which is where the median FFN-down tensor lives.

**Unified CALIB_K:** geometric mean of per-band candidates.

**Cross-class divergence guard.** If the per-band candidate `K` values span more than 3× (ratio of max candidate to min candidate > 3), the calibration binary exits non-zero with a diagnostic listing the per-band candidates. A 3× spread is evidence the formula isn't well-calibrated for cross-class no-weights-vs-weights agreement; a threshold retune can't fix this and a deeper finding is needed (likely Phase 2 territory). The guard prevents silent papering-over.

**Expected spread.** Within ~2× across bands, because the formula's shape-dependence is driven primarily by `numel` and fan-in ratios, which are bounded across transformer-style shapes.

### 3.3 Post-retune distribution (calib_small, 74 tensors)

| Tier | Source | Count |
|---|---|---|
| High | Kind-overridden (embed, norms, first/last) | 32 |
| High | attn QKVO (formula-driven, reinforces override intuition) | 24 |
| Medium | ffn_gate + ffn_up | 12 |
| Low | ffn_down | 6 |
| VeryLow | (none on this corpus) | 0 |

VeryLow stays in the enum as the fallback for out-of-band generics (heavily-pruned biases, unusual shapes); rare by design, which matches the tier's name better than "generic default."

### 3.4 Post-retune invariant verification

All nine invariants from `project_cpdt_weight_aware_invariants.md` stay satisfied:

- **#1 ANALYSIS_VERSION bump:** `ANALYSIS_VERSION` → 2 in Commit 2, same commit as the threshold change.
- **#2 joint-solver reads-only-aggregates:** untouched.
- **#3 kind-override precedence:** untouched; only the generic branch's scored values change.
- **#4 scorer is concrete:** untouched.
- **#5 no rank normalization:** untouched.
- **#6 --weights format override:** untouched.
- **#7 <5% disagreement:** preserved by recalibrated `CALIB_K`; the regenerated `cpdt_sensitivity_disagreement` gate passes at the new values.
- **#8 weights-present byte-identity:** regenerate `expected_weights_present.json` from the new scorer output; this IS the update, not a break.
- **#9 scorer state from `WeightMap` only:** untouched.

## 4. Commit Sequencing

### 4.1 Commit 1 — `feat(cpdt): compute threshold + CALIB_K values from fixture score bands`

Files:
- `crates/nsl-codegen/src/bin/cpdt_calibrate.rs` — add score-band analysis + threshold computation + CALIB_K computation, behind a new `--emit-calibration` flag (default behavior unchanged).
- `crates/nsl-codegen/src/bin/cpdt_fixture_generate.rs` — add `calib_medium` generation to the `target/cpdt_calibration/` path (not committed; regen-at-test-time).

**Acceptance:** `cargo run --features calibrate --bin cpdt_calibrate -- tests/fixtures/cpdt_calibration/ --emit-calibration` runs, emits the constants block + the CALIB_K block, exits 0. The scorer's in-code constants and behavior are unchanged; `baseline_heuristic.json` and `expected_weights_present.json` are byte-identical before/after the commit. No test files change.

### 4.2 Commit 2 — `feat(cpdt): retune T0/T1/T2 + CALIB_K + bump ANALYSIS_VERSION`

Files:
- `crates/nsl-codegen/src/cpdt_sensitivity.rs` — apply the new `T0`, `T1`, `T2`, `CALIB_K` constants (emitted from Commit 1's binary); bump `ANALYSIS_VERSION` from 1 to 2.
- `tests/fixtures/cpdt_calibration/baseline_heuristic.json` — regenerated.
- `tests/fixtures/cpdt_calibration/expected_weights_present.json` — regenerated.
- `crates/nsl-codegen/tests/cpdt_tier_agreement.rs` — update `tier_agreement_full_on_calib_small_by_construction` to assert the new non-degenerate distribution. The test's existing comment already anticipates this: *"calibration currently produces identical tier sets; recalibration will break this — update test when threshold tuning lands."*
- `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md` — append a new section: *"Phase 1 threshold retune — 2026-04-19 — calibration correction."*

**Acceptance — name the specific assertions that move:**
1. `ANALYSIS_VERSION == 2` (was 1).
2. `cpdt_sensitivity_snapshot::weights_present_matches_expected_snapshot` passes against the regenerated `expected_weights_present.json`.
3. `cpdt_sensitivity_disagreement::weighted_disagreement_below_5_percent` passes under the new `CALIB_K` at the 5% gate (unchanged).
4. `cpdt_tier_agreement::tier_agreement_full_on_calib_small_by_construction` — **specific assertion updates:** replace the `agree_l == total_l` check with a tier-distribution non-degeneracy check asserting that each of `{High, Medium, Low}` has at least one member on `calib_small` (VeryLow intentionally allowed to be zero by §3.3 design).
5. `cpdt_sensitivity_adversarial::adversarial_localized_tier_shift` still passes: the `M = CALIB_T0 / s_pre * 1.5` multiplier scales with the new `T0`, so the target still hits `Tier::High`.
6. `cpdt_sensitivity_primitives` — all 22 boundary tests still pass (they use `CALIB_T0 + 1e-6` / `CALIB_T2 - 1e-6` patterns that are T-value-agnostic).

## 5. Retrospective Addendum

Location: **append** to `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md` as a new top-level section.

Title: `## Phase 1 threshold retune — 2026-04-19 — calibration correction`

(The word "correction" signals at a glance this is a post-ship fix, not an evolution of the original design.)

Body captures:
1. The empirical observation (degenerate binary distribution: 32 High + 42 VeryLow, 0 Medium, 0 Low on `calib_small`).
2. Why the retune preserves the calibration story (formula unchanged, invariant #7 preserved via `CALIB_K` recalibration, not by weakening the gate).
3. Per-band score observations on `calib_small` and the computed `T0/T1/T2/CALIB_K` values.
4. The post-retune distribution.
5. The new institutional rule (§6 below).

## 6. Institutional Discipline — Tier-Distribution Non-Degeneracy Check

Addendum for future Phase 1-style tier-assignment work. Adds to the emerging family of institutional close-out rules (the same family that includes the WRGA B.3.1 "test at the scale you target" rule and the PCA Tier A convention-match rule). Not formally numbered here; the retrospective addendum in phase2-stub.md is free to assign an index if the rule family gets codified.

**Rule:** any tier-assignment system that ships an N-tier scheme MUST include a close-out gate that asserts the **primary** tier distribution is non-degenerate on the calibration corpus. Primary-tier set defined as:

> *All tiers the scheme uses except any documented-as-rare fallback tier.*

Non-degeneracy defined as:

> *Every primary tier has at least one tensor assigned to it on at least one calibration fixture.*

**Fallback-tier exemption:** if a scheme documents a specific tier as an intentionally-rare fallback (e.g. CPDT's VeryLow for out-of-band generics), that tier is exempt from the non-degeneracy check. The exemption must be named explicitly in the spec (not left implicit) and must name the conditions under which the tier is expected to fire.

**Why this rule exists:** Phase 1 shipped a 4-tier scheme that operated as a 2-tier scheme in practice. Both Phase 1 close-out gates (byte-identity on no-weights, <5% disagreement) passed trivially because both paths produced the same degenerate distribution. The degeneracy slipped past close-out and surfaced in post-ship inspection.

**How to apply:** add a tier-count assertion to the close-out test suite alongside byte-identity and disagreement gates. Count tiers in the post-refactor snapshot JSONs; assert each primary tier's count > 0 on at least one fixture.

**Scope:** applies to any future tier-assignment scheme in NSL (CPDT, WRGA tiering, CEP importance buckets, FASE precision stages). The cost is trivial (one test); the signal is catching degenerate distributions at close-out rather than at first actual use.

**CPDT's application:** primary tiers are `{High, Medium, Low}`. VeryLow is the documented fallback per §3.3 ("rare by design; fallback for out-of-band generics"). Commit 2's updated `tier_agreement_full_on_calib_small_by_construction` asserts the primary-tier non-degeneracy on `calib_small`.

## 7. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Per-band CALIB_K candidates span >3× | Low | Guard exits non-zero with diagnostic; escalate to Phase 2 investigation rather than ship. |
| Cross-fixture band ordering differs | Very Low | Guard exits non-zero with diagnostic; pick the narrower fixture set or revisit shape classification. |
| calib_tiny's L=2 means every layer is first-or-last → every tensor kind-overridden → no generic samples to contribute to bands | Certain (by design) | Compute bands from `calib_small` and `calib_medium` only; use `calib_tiny` only for the consistency guard (its generic count is 1, contributes to spread measurement). |
| Regenerated JSONs introduce CRLF noise on Windows | Low | `.gitattributes` + explicit `--text-format-json` write; existing pattern from Phase 1. |
| Adversarial test's `M = CALIB_T0 / s_pre × 1.5` still produces `Tier::High` under new `T0` | Certain (by algebra) | `M` is computed from the live `T0`; scales automatically. |

## 8. Close-Out Criteria

- Commit 1 acceptance (§4.1) satisfied: binary emits constants, scorer behavior unchanged.
- Commit 2 acceptance (§4.2) satisfied: constants applied, JSONs regenerated, 6 named assertions verified.
- `ANALYSIS_VERSION = 2` in source.
- Retrospective addendum committed to `phase2-stub.md` under the titled section.
- All nine Phase 1 invariants (project memory file) still hold; verify in the Commit 2 message.
- MEMORY.md entry updated in session memory directory to reflect the post-retune state and the new institutional rule.

## 9. Summary

Phase 1 shipped with placeholder `T0/T1/T2/CALIB_K` values that produced a degenerate binary distribution in practice. This spec corrects that via a two-commit retune: Commit 1 adds the threshold-computation capability to the dev-only calibration binary (no behavior change); Commit 2 applies the emitted constants, bumps `ANALYSIS_VERSION` to 2, regenerates both snapshot JSONs, and updates the `tier_agreement_full_on_calib_small_by_construction` canary to a non-degeneracy assertion. `CALIB_K` is recalibrated in the same commit to preserve invariant #7's 5% disagreement gate without weakening it. A new Appendix-B-style institutional rule — tier-distribution non-degeneracy as a close-out gate — is added to prevent the class of omission that let the degenerate distribution ship.
